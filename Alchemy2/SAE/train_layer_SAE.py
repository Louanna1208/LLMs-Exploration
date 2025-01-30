import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from ast import literal_eval
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models import SparseAutoEncoder
import os
import glob
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train the logistic regression model
    logistic_regression_model = LogisticRegression(max_iter=10000)
    logistic_regression_model.fit(X_train, y_train)
    #return only the test accuracy
    return logistic_regression_model.score(X_test, y_test)
    

def train_layer_SAE(model_size,layer):
    # change the working directory to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Model parameters
    config = {}
    config['hidden_size'] = 8192 # Dimension of the latent space
    config['top_k'] = 100
    config['NONLINEARITY'] = "ReLU"
    config['lambda'] = 1e-6
    config['max_lambda'] = 1e-6
    config['num_epochs'] = 150
    config['lr'] = 1e-4
    config['batch_size'] = 256
    config['dtype'] = torch.bfloat16

    # load embeddings and labels
    text_embedding = np.load(os.path.join(script_dir, f'../embedding_result/dataset/Layer_embedding/LLaMA3_{model_size}_embedding_layer{layer}.npy'))

    # reshape the text_embedding to be 2D
    text_embedding = text_embedding.reshape(text_embedding.shape[0], -1)
    print(f"text_embedding shape: {text_embedding.shape}")
    # Convert text_embedding to float32
    text_embedding = text_embedding.astype(np.float32)
    config['input_size'] = text_embedding.shape[1]
    # normalize the text_embedding to expected L2 norm
    text_embedding, _ = normalize_dataset_to_expected_l2(torch.tensor(text_embedding), config['input_size'],config['dtype'])

    # Prepare the dataset
    dataset = TensorDataset(text_embedding)

    # Split the dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    # Instantiate VAE model with classifier, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = SparseAutoEncoder(config).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=config['lr'])

    # Set number of epochs

    reconstruction_losses_list = []
    l1_penalties_list = []

    # Training loop in tqdm
    for epoch in range(config['num_epochs']):
        sae_model.train()  # Set model to training mode
        epoch_loss = 0
        reconstruction_losses = []
        l1_penalties = []
        # gradually increase the lamda as the epoch increases
        config['lambda'] = min(config['max_lambda'], config['lambda']  + 1e-6 )
        for batch_idx, (inputs,) in enumerate(train_loader):
            inputs = inputs.to(device)  # Move input data to GPU if available
        
            optimizer.zero_grad()  # Reset gradients
        
            # Forward pass
            reconstruction_loss, l1_penalty, reconstructed_data, encoded_features = sae_model(inputs)
        
            # save the reconstruction loss and l1 penalty
            reconstruction_losses.append(reconstruction_loss.item())
            l1_penalties.append(l1_penalty.item())
            # Calculate total loss (reconstruction + sparsity penalty)
            total_loss = reconstruction_loss + config['lambda'] * l1_penalty  # You can tune the weight (0.1) for L1 penalty

            # Backpropagation
            total_loss.backward()

            # Update model weights
            optimizer.step()
    
            epoch_loss += total_loss.item()

            # sae_model.normalize_weights()
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {epoch_loss/len(train_loader):.4f}')
        #calculate the mean of the reconstruction loss and l1 penalty and save them in a list   
        reconstruction_losses = np.mean(reconstruction_losses)
        l1_penalties = np.mean(l1_penalties)
        reconstruction_losses_list.append(reconstruction_losses)
        l1_penalties_list.append(l1_penalties)

    # evaluate the model
    sae_model.eval()
    with torch.no_grad():
        test_reconstruction_loss = 0
        test_l1_penalty = 0
        for batch_idx, (inputs,) in enumerate(test_loader):
            inputs = inputs.to(device)
            reconstruction_loss, l1_penalty, _, _ = sae_model(inputs)
            test_reconstruction_loss += reconstruction_loss.item()
            test_l1_penalty += l1_penalty.item()

    all_inputs = text_embedding.to(device)
    sparsity_ratio = calculate_sparsity(sae_model, all_inputs, 1e-4)
    print(f'Sparsity: {torch.mean(sparsity_ratio):.4f}')

    # set the whole dataset as input and move to the model
    inputs = text_embedding.to(device)
    sparsity_ratio = calculate_sparsity(sae_model, inputs, 1e-4)
    sparsity_neuron = torch.mean(sparsity_ratio, dim=0)
    sparsity_ratio = torch.mean(sparsity_ratio)
    print(f'Sparsity: {sparsity_ratio:.4f}')
    sparsity_ratio = sparsity_ratio.cpu().numpy()
    #plot histogram of the sparsity neuron
    sparsity_neuron = sparsity_neuron.cpu().numpy()

    # save the model
    folder_name = f'../layer_SAE_model/dataset'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    torch.save(sae_model.state_dict(), f'{folder_name}/SAE_model_{model_size}_layer{layer}.pth')

    # train the input embedding a logistic regression model for choices
    # load the choice data
    # choice_data = load_choice_data(dataset_type)
    # train the logistic regression model
    # with torch.no_grad():
    #     _, _, reconstructed_embedding, _ = sae_model(inputs)
    # #convert the reconstructed_embedding tensor to float32
    # reconstructed_embedding = reconstructed_embedding.to(torch.float32)
    # reconstructed_embedding = reconstructed_embedding.cpu().numpy()
    # # covert text_embedding tensor BF16 to float32 and numpy
    # text_embedding = text_embedding.to(torch.float32)
    # text_embedding = text_embedding.cpu().numpy()

    # # train the logistic regression model with train test split
    # original_test_accuracy = train_logistic_regression(text_embedding, choice_data)
    # reconstructed_test_accuracy = train_logistic_regression(reconstructed_embedding, choice_data)
    return test_reconstruction_loss / len(test_loader), test_l1_penalty / len(test_loader), sparsity_neuron, sparsity_ratio

def visualize_training_log(model_size, log_df):
    # visualize how the test reconstruction loss, test l1 penalty, sparsity neuron, sparsity ratio, original test accuracy, and reconstructed test accuracy change as the layer increases in spearate figures
    # save the figures in the ../layer_SAE_model/{dataset_type}_dataset/SAE_training_log_{model_size}/pictures
    if not os.path.exists(f'../layer_SAE_model/dataset/SAE_training_log_{model_size}/pictures'):
        os.makedirs(f'../layer_SAE_model/dataset/SAE_training_log_{model_size}/pictures')
    # plot the test reconstruction loss
    plt.plot(log_df['Layer'], log_df['Test Reconstruction Loss'], label='Test Reconstruction Loss')
    plt.title('Test Reconstruction Loss')
    plt.xlabel('Layer')
    plt.ylabel('Test Reconstruction Loss')
    plt.savefig(f'../layer_SAE_model/dataset/SAE_training_log_{model_size}/pictures/test_reconstruction_loss.png',dpi=300)
    plt.close()
    
    #plot the test l1 penalty
    plt.plot(log_df['Layer'], log_df['Test L1 Penalty'], label='Test L1 Penalty')
    plt.title('Test L1 Penalty')
    plt.xlabel('Layer')
    plt.ylabel('Test L1 Penalty')
    plt.savefig(f'../layer_SAE_model/dataset/SAE_training_log_{model_size}/pictures/test_l1_penalty.png',dpi=300)
    plt.close()

    # plot the sparsity ratio
    plt.plot(log_df['Layer'], log_df['Sparsity Ratio'], label='Sparsity Ratio')
    plt.title('Sparsity Ratio')
    plt.xlabel('Layer')
    plt.ylabel('Sparsity Ratio')
    plt.savefig(f'../layer_SAE_model/dataset/SAE_training_log_{model_size}/pictures/sparsity_ratio.png',dpi=300)
    plt.close()

    # # plot the original test accuracy and reconstructed test accuracy on the same figure with different colors
    # plt.plot(log_df['Layer'], log_df['Original Test Accuracy'], label='Original Test Accuracy')
    # plt.plot(log_df['Layer'], log_df['Reconstructed Test Accuracy'], label='Reconstructed Test Accuracy')
    # plt.title('Test Accuracy')
    # plt.xlabel('Layer')
    # plt.ylabel('Test Accuracy')
    # plt.legend()
    # plt.savefig(f'../layer_SAE_model/{dataset_type}_dataset/SAE_training_log_{model_size}/pictures/test_accuracy.png',dpi=300)
    # plt.close()


def train_all_layers_SAE(model_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #search all the layers files to determine the number of layers
    layer_files = glob.glob(os.path.join(script_dir, f'../embedding_result/dataset/Layer_embedding/LLaMA3_{model_size}_embedding_layer*.npy'))
    layer_files.sort()
    num_layers = len(layer_files)
    # initalize a dataframe to store the log of the training
    log_df = pd.DataFrame(columns=['Layer', 'Test Reconstruction Loss', 'Test L1 Penalty', 'Sparsity Neuron', 'Sparsity Ratio'])
    for layer in tqdm(range(num_layers), desc="Training Layers:"):
        test_reconstruction_loss, test_l1_penalty, sparsity_neuron, sparsity_ratio = train_layer_SAE(model_size, layer)
        test_reconstruction_loss = np.float32(test_reconstruction_loss)
        test_l1_penalty = np.float32(test_l1_penalty)
        sparsity_neuron = np.float32(sparsity_neuron)
        sparsity_ratio = np.float32(sparsity_ratio)
        print(f'Layer {layer}: Test Reconstruction Loss: {test_reconstruction_loss:.4f}, Test L1 Penalty: {test_l1_penalty:.4f}, Sparsity Ratio: {sparsity_ratio:.4f}')
        # add the log to the dataframe
        new_row = pd.DataFrame({
            'Layer': [layer], 
            'Test Reconstruction Loss': [test_reconstruction_loss], 
            'Test L1 Penalty': [test_l1_penalty], 
            'Sparsity Ratio': [sparsity_ratio]
            # 'Original Test Accuracy': [original_test_accuracy],
            # 'Reconstructed Test Accuracy': [reconstructed_test_accuracy]
        })
        log_df = pd.concat([log_df, new_row], ignore_index=True)
    # save the dataframe to a csv file
    log_df.to_csv(os.path.join(script_dir, f'../layer_SAE_model/dataset/SAE_training_log_{model_size}.csv'), index=False)
    visualize_training_log(model_size, log_df)


def main():
    parser = argparse.ArgumentParser(description='Run LLaMA prediction')
    parser.add_argument("--model_size", type=str, default="7b", help="Specify the model size, e.g., '7b', '8b', '13b', '70b'")
    
    args = parser.parse_args()
    model_size = args.model_size

    # Run experiment
    train_all_layers_SAE(model_size)

if __name__ == "__main__":
    main()
