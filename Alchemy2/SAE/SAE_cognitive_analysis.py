from models import SparseAutoEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import json
import pickle
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import argparse
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_embeddings(size,layer):
    return np.load(f'../embedding_result/dataset/Layer_embedding/LLaMA3_{size}_embedding_layer{layer}.npy')


def load_behavioral_and_model_data():
    # Choose the dataset file based on the 'dataset' argument
    csv_file = '../output/data/Llama3_70B_element_value.csv'

    df = pd.read_csv(csv_file)
    df.columns = ['id', 'trial', 'element', 'choice', 'cbu_value', 'cbv_value', 'recency_value', 'empowerment_value']
    return df

def normalize_dataset_to_expected_l2(X, n, dtype):
    """
    Normalize the dataset so that the expected L2 norm of the data points equals sqrt(n).
    
    Args:
    - X (torch.Tensor): The input dataset, with shape [num_samples, input_size].
    - n (float): The target value for the expected squared L2 norm (typically input_size).
    
    Returns:
    - X_normalized (torch.Tensor): The normalized dataset.
    - scaling_factor (float): The factor used to scale the dataset.
    """
    # Step 1: Compute the L2 norms of all data points
    l2_norms = torch.norm(X, p=2, dim=1)
    print(l2_norms.shape)
    # Step 2: Compute the mean L2 norm
    mean_l2_norm = torch.mean(l2_norms)

    # Step 3: Compute the scaling factor to adjust the mean L2 norm to sqrt(n)
    scaling_factor = torch.sqrt(torch.tensor(n, dtype=X.dtype, device=X.device)) / mean_l2_norm

    # Step 4: Normalize the dataset by the scaling factor
    X_normalized = X * scaling_factor

    # turn x_normalized to bfloat16
    X_normalized = X_normalized.to(dtype)
    return X_normalized, scaling_factor

def calculate_sparsity(model, input_data, threshold=1e-6):
    """
    Calculates the sparsity ratio in the latent space of a trained sparse autoencoder.
    
    Parameters:
    - model: The trained autoencoder model with an encoder.
    - input_data: The input data to pass through the encoder.
    - threshold: The threshold below which an activation is considered "zero" (sparse).
    
    Returns:
    - sparsity_ratio: The proportion of activations in the latent space below the threshold.
    """
    # Pass the input through the encoder to get the latent activations
    with torch.no_grad():  # No need to calculate gradients
        _, _, _, latent_activations = model(input_data)
    
    # Calculate the proportion of latent activations below the threshold (sparse)
    sparsity_ratio = (latent_activations.abs() < threshold).float()
    
    return sparsity_ratio

def train_logistic_regression(X,y):
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2024)
    # train the logistic regression model
    logistic_regression_model = LogisticRegression(max_iter=10000)
    logistic_regression_model.fit(X_train, y_train)
    # predict the test data
    y_pred = logistic_regression_model.predict(X_test)
    #return only the test accuracy
    return accuracy_score(y_test, y_pred)

def get_max_correlation_result(latent_space, metrics, type = 'correlation'):
    #for each column of latent space, do correlation or regression with the metrics, record the max correlation coefficient and the corresponding neuron index
    correlation_result = []
    correlation_neuron = []
    # find the nan values in the metrics and remove them, remove the corresponding rows in latent_space
    nan_indices = np.where(np.isnan(metrics))[0]
    latent_space = np.delete(latent_space, nan_indices, axis=0)
    metrics = np.delete(metrics, nan_indices, axis=0)
    print(f'latent_space shape: {latent_space.shape}')
    print(f'metrics shape: {metrics.shape}')
    if latent_space.shape[1] == 0:
        corr_result = 0
        i = 0
        correlation_neuron.append(i)
        correlation_result.append(corr_result)
    else:
        for i in range(latent_space.shape[1]):
            if type == 'correlation':    
                corr_result = np.corrcoef(latent_space[:, i], metrics)[0,1]
            elif type == 'logistic':
                latent_space_selected = latent_space[:, i].reshape(-1, 1)
                # do logistic regression
                log_reg = LogisticRegression()
                # normalize the latent_space_selected
                latent_space_selected = (latent_space_selected - np.mean(latent_space_selected)) / np.std(latent_space_selected)
                log_reg.fit(latent_space_selected, metrics)
                corr_result = log_reg.coef_[0][0]
            correlation_result.append(corr_result)
            correlation_neuron.append(i)
    # find out the max absolute correlation coefficient and the corresponding neuron index
    max_correlation_index = np.argmax(correlation_result)
    max_correlation = correlation_result[max_correlation_index]
    max_correlation_neuron = correlation_neuron[max_correlation_index]
    return max_correlation, max_correlation_neuron

def find_max_correlation_neuron(size,layer):
    # Model parameters
    config = {}
    config['hidden_size'] = 8192 # Dimension of the latent space
    config['top_k'] = 100
    config['NONLINEARITY'] = "ReLU"
    config['lambda'] = 1e-6
    config['max_lambda'] = 5e-5
    config['num_epochs'] = 150
    config['lr'] = 5e-4
    config['batch_size'] = 64
    config['dtype'] = torch.bfloat16

    text_embedding = load_embeddings(size,layer)
    config['input_size'] = text_embedding.shape[1]

    # Instantiate VAE model with classifier, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = SparseAutoEncoder(config).to(device)
    # load the trained model
    sae_model.load_state_dict(torch.load(f'../layer_SAE_model/dataset/SAE_model_{size}_layer{layer}.pth'))

    # load behavioral and model data
    behavioral_data = load_behavioral_and_model_data()

    # Ensure the length of behavioral_data matches text_embedding rows
    # If not, you need to handle indexing or filtering so that they match.
    # For this example, we assume they match in order.
    if behavioral_data.shape[0] != text_embedding.shape[0]:
        raise ValueError(f"Mismatch between number of trials in behavioral data {behavioral_data.shape[0]} and embeddings {text_embedding.shape[0]}.")

    # normalize the text_embedding to expected L2 norm
    text_embedding, scaling_factor = normalize_dataset_to_expected_l2(torch.tensor(text_embedding), config['input_size'], config['dtype'])
    print(f'text_embedding shape: {text_embedding.shape}')

    # get the latent space for all the data points
    inputs = text_embedding.to(device)
    with torch.no_grad():
        _, _, reconstructed_embedding,latent_space = sae_model(inputs)

    # convert latent_space to float32
    latent_space = latent_space.to(torch.float32)
    latent_space = latent_space.cpu().numpy()

    # calculate the sparsity ratio in the latent space
    sparsity_ratio = calculate_sparsity(sae_model, inputs, 1e-4)
    sparsity_neuron = torch.mean(sparsity_ratio, dim=0)

    reconstructed_embedding = reconstructed_embedding.to(torch.float32)
    reconstructed_embedding = reconstructed_embedding.cpu().numpy()

    text_embedding = text_embedding.to(torch.float32)
    text_embedding = text_embedding.cpu().numpy()

    #plot histogram of the sparsity neuron
    sparsity_neuron = sparsity_neuron.cpu().numpy()

    # Optional: filter neurons based on sparsity if needed
    activation_neuron_index = np.where(sparsity_neuron < 1)[0]
    activation_latent_space = latent_space[:, activation_neuron_index]
    print(f'activation_latent_space shape: {activation_latent_space.shape}')

    # Extract metrics from entire dataset
    choices = behavioral_data['choice'].values
    cbu_values = behavioral_data['cbu_value'].values
    cbv_values = behavioral_data['cbv_value'].values
    recency_values = behavioral_data['recency_value'].values
    empowerment_values = behavioral_data['empowerment_value'].values

    original_test_accuracy = train_logistic_regression(text_embedding, choices)
    reconstructed_test_accuracy = train_logistic_regression(reconstructed_embedding, choices)

    metrics = {
        'choices': choices,
        'cbu_value': cbu_values,
        'cbv_value': cbv_values,
        'recency_value': recency_values,
        'empowerment_value': empowerment_values
    }

    max_correlation_result = {}
    # get the max correlation neuron
    for metric in metrics.keys():
        max_correlation_result[metric] = {}
        print(f'metric: {metric}')
        if metric == 'choices':
            # choices is a binary variable, so use logistic regression
            max_correlation, max_correlation_neuron = get_max_correlation_result(activation_latent_space, metrics[metric], type = 'logistic')
        else:
            max_correlation, max_correlation_neuron = get_max_correlation_result(activation_latent_space, metrics[metric], type = 'correlation')

        max_correlation_result[metric]['correlation'] = max_correlation
        max_correlation_result[metric]['neuron'] = max_correlation_neuron
    
    return max_correlation_result, original_test_accuracy, reconstructed_test_accuracy

def plot_max_correlation_by_metric(max_correlation_result, figure_save_path):
    metrics = list(max_correlation_result[0].keys())
    for metric in metrics:
        layers = list(max_correlation_result.keys())
        max_correlation_list = []
        for layer in layers:
            max_correlation = max_correlation_result[layer][metric]['correlation']
            max_correlation_list.append(max_correlation)
        
        plt.plot(layers, max_correlation_list, label=metric, color = 'pink')
        # use a vertical line to indicate which layer has the max correlation
        max_correlation_layer = layers[max_correlation_list.index(max(max_correlation_list))]
        figure_save_path_metric = figure_save_path + f'_{metric}.png'
        plt.axvline(x=max_correlation_layer, color='red', linestyle='--', label=f'Max Correlation Layer: {max_correlation_layer}')
        plt.title(f'Max Correlation of {metric} by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Max Correlation')
        plt.legend()
        plt.savefig(figure_save_path_metric, dpi=300)
        plt.close()

def plot_test_accuracy(layer_original_test_accuracy, layer_reconstructed_test_accuracy, figure_save_path):
    # plot the original and reconstructed test accuracy
    layers = list(layer_original_test_accuracy.keys())
    original_test_accuracy_list = []
    reconstructed_test_accuracy_list = []
    for layer in layers:
        original_test_accuracy_list.append(layer_original_test_accuracy[layer])
        reconstructed_test_accuracy_list.append(layer_reconstructed_test_accuracy[layer])
    plt.plot(layers, original_test_accuracy_list, label = 'Original')
    plt.plot(layers, reconstructed_test_accuracy_list, label = 'Reconstructed')
    plt.xlabel('Layer')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig(figure_save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Find Max Correlation Neuron')
    parser.add_argument("--model_size", type=str, default="70B", help="Specify the model size, e.g., '7B', '8B', '13B', '70B'")
    
    args = parser.parse_args()
    model_size = args.model_size

    script_dir = os.path.dirname(os.path.abspath(__file__))
    #search all the layers files to determine the number of layers
    layer_files = glob.glob(os.path.join(script_dir, f'../embedding_result/dataset/Layer_embedding/LLaMA3_{model_size}_embedding_layer*.npy'))
    save_result_path = os.path.join(script_dir, f'../embedding_result/dataset/max_correlation_LLaMA3_{model_size}.pkl')
    figure_save_folder_path = os.path.join(script_dir, f'../pic/SAE_cognitive_correlation/dataset')
    if not os.path.exists(figure_save_folder_path):
        os.makedirs(figure_save_folder_path)
    figure_save_path = os.path.join(figure_save_folder_path, f'LLaMA3_{model_size}')
    layer_files.sort()
    num_layers = len(layer_files)
    layer_max_correlation_result = {}
    layer_original_test_accuracy = {}
    layer_reconstructed_test_accuracy = {}
    # Run experiment
    for layer in tqdm(range(num_layers), desc='Processing layers'):
        max_correlation_result, original_test_accuracy, reconstructed_test_accuracy = find_max_correlation_neuron(model_size, layer)
        layer_max_correlation_result[layer] = max_correlation_result
        layer_original_test_accuracy[layer] = original_test_accuracy
        layer_reconstructed_test_accuracy[layer] = reconstructed_test_accuracy
    # save the result
    with open(save_result_path, 'wb') as file:
        pickle.dump(layer_max_correlation_result, file)
    # plot the max correlation by metric
    plot_max_correlation_by_metric(layer_max_correlation_result, figure_save_path)
    # save the original and reconstructed test accuracy as one csv
    original_test_accuracy_df = pd.DataFrame(layer_original_test_accuracy, index = [0])
    reconstructed_test_accuracy_df = pd.DataFrame(layer_reconstructed_test_accuracy, index = [0])
    test_accuracy_df = pd.concat([original_test_accuracy_df, reconstructed_test_accuracy_df], axis = 1)
    test_accuracy_df.to_csv(os.path.join(figure_save_folder_path, f'LLaMA3_{model_size}_test_accuracy.csv'), index = False)
    # plot the original and reconstructed test accuracy
    figure_save_path_test_accuracy = os.path.join(figure_save_folder_path, f'LLaMA3_{model_size}_test_accuracy.png')
    plot_test_accuracy(layer_original_test_accuracy, layer_reconstructed_test_accuracy, figure_save_path_test_accuracy)

if __name__ == "__main__":
    main()



















