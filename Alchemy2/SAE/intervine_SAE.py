from os.path import join
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from torch import cuda
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
import os
import pickle
from models import SparseAutoEncoder
import numpy as np
import csv


# os.environ['TRANSFORMERS_CACHE'] = "E:/Huggingface/model_cache"

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Load LLaMA model and tokenizer
def model_loading(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    API_key = 'api_key'


    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # begin initializing HF items, need auth token for these
    model_config = AutoConfig.from_pretrained(
        model_id,
        token=API_key
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        #device_map={'': 0},
        device_map='auto',
        token=API_key,
        torch_dtype = 'auto'
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=API_key
    )
    return model, tokenizer

def normalize_dataset_to_expected_l2(X, n, dtype):
    """
    Normalize the dataset so that the expected L2 norm of the data points equals sqrt(n).
    
    Args:
    - X (torch.Tensor): The input dataset or single data point
    - n (float): The target value for the expected squared L2 norm (typically input_size)
    - dtype: The desired output dtype
    
    Returns:
    - X_normalized (torch.Tensor): The normalized data
    - scaling_info (dict): Dictionary containing normalization factors:
        - data_norm: The mean L2 norm of the original data
        - target_norm: The target L2 norm (sqrt(n))
    """
    # Step 1: Compute L2 norms (handles both batched and single data points)
    if X.dim() == 1:
        l2_norm = torch.norm(X, p=2)
        mean_l2_norm = l2_norm  # For single data point, norm is the mean
    else:
        l2_norms = torch.norm(X, p=2, dim=1)
        mean_l2_norm = torch.mean(l2_norms)

    # Step 2: Calculate target norm (sqrt(n))
    target_norm = torch.sqrt(torch.tensor(n, dtype=X.dtype, device=X.device))
    
    # Step 3: Compute scaling factor
    scaling_factor = target_norm / mean_l2_norm
    
    # Step 4: Apply normalization
    X_normalized = X * scaling_factor
    X_normalized = X_normalized.to(dtype)
    
    # Return normalized data and scaling components
    scaling_info = {
        "data_norm": mean_l2_norm,
        "target_norm": target_norm,
    }
    
    return X_normalized, scaling_info

def load_correlations(model_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, f'../embedding_result/dataset/max_correlation_LLaMA3_{model_size}.pkl')
    with open(path, 'rb') as file:
        correlations = pickle.load(file)
    return correlations

def get_intervention_location(correlations,variable):
    layers = list(correlations.keys())
    # find maximum correlation layer
    max_correlation_list = []
    for layer in layers:
        max_correlation = correlations[layer][variable]['correlation']
        max_correlation_list.append(max_correlation)
    # get the maximum correlation layer for 'choices'
    max_correlation_layer = layers[max_correlation_list.index(max(max_correlation_list))]
    # get the latent id for the maximum correlation layer
    max_correlation_latent_id = correlations[max_correlation_layer][variable]['neuron']
    return max_correlation_layer, max_correlation_latent_id

# save_all_game_states
def save_all_game_states(result_dict, save_path, model_size, max_trials, variable):
    # Create directories if they don't exist
    file_name = os.path.join(save_path, f'Base/LLaMA_3_{model_size}/intervention_{variable}_{max_trials}_results.json')
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))  # Create directories if they don't exist

    # Save the complete result dictionary to one JSON file
    with open(file_name, 'w') as file:
        json.dump(result_dict, file, indent=4)
    print(f"All game runs saved to {save_path}Base/LLaMA_3_{model_size}/intervention_{variable}_{max_trials}_results.json")

def load_embeddings(size,layer):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return np.load(os.path.join(script_dir, f'../embedding_result/dataset/Layer_embedding/LLaMA3_{size}_embedding_layer{layer}.npy'))

# Define the Game class
def load_game_tree(script_dir):
    # Load Game Tree JSON for Element Mapping
    with open(os.path.join(script_dir, '../dataset/alchemy2Gametree.json'), 'r') as f:
        game_tree = json.load(f)

    # Create a lookup dictionary for element codes to names
    element_code_to_name = {}
    for code, element_data in game_tree.items():
        element_code_to_name[int(code)] = element_data["name"]

    # Load Combination Table
    combination_table = {}
    with open(os.path.join(script_dir, '../dataset/alchemy2CombinationTable.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            first = int(row['first'])
            second = int(row['second'])
            success = int(row['success'])
            result = int(row['result'])

            for key in [(first, second), (second, first)]:  # Add both combinations
                if key not in combination_table:
                    combination_table[key] = []  # Initialize an empty list for each new key
                combination_table[key].append((success, result))
    return element_code_to_name, combination_table

class Game:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.element_code_to_name,self.combination_table = load_game_tree(script_dir)
        self.inventory = {
            self.element_code_to_name[0],  # "water"
            self.element_code_to_name[1],  # "fire"
            self.element_code_to_name[2],  # "earth"
            self.element_code_to_name[3],  # "air"
        }
        self.combination_storage = []
        self.inventory_size = len(self.inventory)

    def check_goals(self):
        self.inventory_size = len(self.inventory)
        condition_elements = []
        if (self.inventory_size >= 50):
            condition_elements.append(35)
        if (self.inventory_size >= 100):
            condition_elements.append(40)
        if (self.inventory_size >= 150):
            condition_elements.append(605)
        if (self.inventory_size >= 150):
            condition_elements.append(606)
        if (self.inventory_size >= 150):
            condition_elements.append(566)
        if (self.inventory_size >= 300):
            condition_elements.append(627)

        for element_code in condition_elements:
            element_name = self.element_code_to_name.get(element_code)
            if element_name and element_name not in self.inventory:
                self.inventory.add(element_name)

    def store_trial_info(self, run, step, combination, success, new_result, result_prompt, log_prob_first, log_prob_second, prompt):
        trial_data = {
            'id': run,
            'trial': step,
            'first': combination[0],
            'second': combination[1],
            'success': success,
            'result': new_result if new_result else -1,
            'result_prompt': result_prompt,
            'inventory': list(self.inventory),
            'log_prob_first': log_prob_first,
            'log_prob_second': log_prob_second,
            'prompt': prompt
        }
        self.combination_storage.append(trial_data)
        return self.combination_storage

    def get_past_trials(self):
        return " \\n ".join([f"Trial {t['trial']}: {t['first']} + {t['second']} -> "
                              f"Success: {t['success']}, Result: {t['result']}. "
                              f"{t['result_prompt']}"
                              for t in self.combination_storage])

    def get_inventory_status(self):
        return ", ".join([f"{element}" for element in self.inventory])

    def check_combination(self, element1, element2):
        result_prompt = ""
        code1 = next((code for code, name in self.element_code_to_name.items() if name == element1), None)
        code2 = next((code for code, name in self.element_code_to_name.items() if name == element2), None)

        if code1 is not None and code2 is not None:
            results = self.combination_table.get((code1, code2))

            if results:
                possible_results = [result_code for success, result_code in results if success == 1]

                for result_code in possible_results:
                    result_name = self.element_code_to_name.get(result_code)
                    if result_name and result_name not in self.inventory:
                        self.inventory.add(result_name)
                        result_prompt = f"You successfully created a new element."
                        return 1, result_name, result_prompt

                if possible_results:
                    last_result_name = self.element_code_to_name.get(possible_results[-1], None)
                    if last_result_name:
                        result_prompt = f"You already created this element before. Please select another element combination."
                        return 1, last_result_name, result_prompt

                if all(success == 0 for success, result_code in results):
                    result_prompt = "Combining these two elements failed. Please select another element combination."
                    return 0, None, result_prompt

        result_prompt = "Combining these two elements failed. Please select another element combination."
        return 0, None, result_prompt

def locate_token_positions_full_prompt(trial_prompt, inventory_elements, tokenizer):
    """
    Locate absolute token positions of inventory elements in the full prompt, considering tokens before Current Inventory.
    """
    # Tokenize the full prompt
    inputs = tokenizer(trial_prompt, return_tensors="pt", truncation=True, add_special_tokens=False)
    full_prompt_token_ids = inputs["input_ids"][0].tolist()

    # Locate the "Current Inventory" substring
    current_inventory_start = trial_prompt.find("Current Inventory:")
    current_inventory_end = trial_prompt.find("\nPast Attempts:")
    current_inventory_text = trial_prompt[current_inventory_start:current_inventory_end]

    # Determine the number of tokens before Current Inventory
    tokens_before_inventory = tokenizer(trial_prompt[:current_inventory_start], return_tensors="pt", truncation=True, add_special_tokens=False)["input_ids"][0].tolist()
    inventory_start_idx = len(tokens_before_inventory)

    # Tokenize the Current Inventory section from the full prompt tokens
    current_inventory_inputs = tokenizer(current_inventory_text, return_tensors="pt", truncation=True, add_special_tokens=False)
    current_inventory_token_ids = current_inventory_inputs["input_ids"][0].tolist()

    # Decode Current Inventory tokens to text
    inventory_text = tokenizer.decode(current_inventory_token_ids)
    #print(f"inventory_text: {inventory_text}")

    # Map elements to their token positions in Current Inventory
    element_positions = {}
    for element in inventory_elements:
        # Tokenize the element
        element = " " + element
        element_inputs = tokenizer(element, return_tensors="pt", truncation=True, add_special_tokens=False)
        element_token_ids = element_inputs["input_ids"][0].tolist()
        #print(f"Element: {element}, Token IDs: {element_token_ids}")

        # Find the token positions of the element in Current Inventory tokens
        for i in range(len(current_inventory_token_ids) - len(element_token_ids) + 1):
            if current_inventory_token_ids[i:i + len(element_token_ids)] == element_token_ids:
                # Calculate absolute positions in the full prompt
                full_prompt_positions = [inventory_start_idx + i + idx for idx in range(len(element_token_ids))]
                element_positions[element] = full_prompt_positions
                break  # Stop after finding the first match

    return element_positions
   
def preapre_intervention(model_size):    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = f'meta-llama/Meta-Llama-3.1-{model_size}-Instruct'

    torch.set_grad_enabled(False)

    global model, tokenizer

    model, tokenizer = model_loading(model_name)

    correlations = load_correlations(model_size)
    intervine_layer, intervine_latent_id = get_intervention_location(correlations,variable)

    # load the SAE modelconfig = {}
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

    text_embedding = load_embeddings(model_size,intervine_layer)
    text_embedding = torch.tensor(text_embedding, dtype=config['dtype'])
    config['input_size'] = text_embedding.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = SparseAutoEncoder(config).to(device)
    # load the trained model
    sae_model.load_state_dict(torch.load(os.path.join(script_dir, f'../layer_SAE_model/dataset/SAE_model_{model_size}_layer{intervine_layer}.pth')))

    _, scaling_factor = normalize_dataset_to_expected_l2(text_embedding, config['input_size'], config['dtype'])
    return model, tokenizer, sae_model, scaling_factor, intervine_layer, intervine_latent_id

def intervention_hook(module, input, output):
    """
    Forward hook to modify hidden states dynamically during forward passes.
    Handles both the initial sequence and sliding context during generation.
    """
    global current_window_start
    # print("Inside forward hook...")
    try:
        hidden_states = output[0]  # Hidden states tensor
        seq_len = hidden_states.size(1)
        # print(f"Hidden states shape: {hidden_states.shape}")

        # Determine positions to modify
        if seq_len > 1:  # Initial forward pass
            # print(f"Sequence length: {seq_len}")
            target_positions = dynamic_target_positions
        else:  # Sliding context during generation
            target_positions = update_positions_in_context(dynamic_target_positions, seq_len, current_window_start)
            current_window_start += 1  # Increment sliding window

        # print(f"Target positions: {target_positions}")

        if not target_positions:
            # print("No valid target positions for the current sequence length. Skipping intervention.")
            return output

        # print(f"Hidden states at target positions before modification: {hidden_states[:, target_positions, :]}")

        # Modify the hidden states at target positions
        hidden_state = hidden_states[:, target_positions, :]
        hidden_state = hidden_state / scaling_factor["data_norm"] * scaling_factor["target_norm"]
        latent_states = sae_model.encode(hidden_state)
        latent_states[:, :, intervine_latent_id] *= intervention_factor
        hidden_reconstruct = sae_model.decode(latent_states)
        hidden_reconstruct = hidden_reconstruct / scaling_factor["target_norm"] * scaling_factor["data_norm"]
        hidden_states[:, target_positions, :] = hidden_reconstruct

        # print(f"Hidden states at target positions after modification: {hidden_states[:, target_positions, :]}")

    except Exception as e:
        print(f"Error during forward hook: {e}")
        raise
    return output


def update_positions_in_context(target_positions, seq_len, sliding_window_start):
    """
    Update target positions to align with the sliding window context.
    """
    updated_positions = [
        pos - sliding_window_start for pos in target_positions
        if sliding_window_start <= pos < sliding_window_start + seq_len
    ]
    return updated_positions

def run_intervention(model, tokenizer, sae_model, scaling_factor, intervine_layer, intervine_latent_id, 
                     system_prompt, trial_prompt, inventory, intervention_factor, 
                     save_result=True, save_hidden_state=False):
                     
    global current_window_start, generated_tokens
    current_window_start = 0  # Tracks the start of the sliding window
    generated_tokens = []     # Tracks tokens generated during generation   
    # Prepare the prompt
    prompt = system_prompt + "\n" + trial_prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_length = inputs["input_ids"].size(1)

    global dynamic_target_positions
    intervention_locs = list(locate_token_positions_full_prompt(prompt, inventory, tokenizer).values())
    dynamic_target_positions = np.concatenate(intervention_locs).tolist()

    if not dynamic_target_positions:
        print("Warning: No valid target positions found.")
        return None, None

    # print(f"Dynamic target positions: {dynamic_target_positions}")

    max_tokens = 3
    model_response = None
    full_hidden_state = None
    # Attach the hook
    try:
        hook = model.model.layers[intervine_layer].register_forward_hook(intervention_hook)
        if save_result:
            # Generate tokens and track them
            outputs = model.generate(
                **inputs,
            max_new_tokens=max_tokens,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )

            # Decode and return the response
            generated_tokens = outputs.sequences[:, -max_tokens:] 
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            # get the first and the last token's log likelihood in generated_tokens
            first_token_log_prob = transition_scores[0][0].item()
            last_token_log_prob = transition_scores[0][2].item()
            model_response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            # print("Decoded model output (new tokens only):", model_response)

        if save_hidden_state:
            # Get the last hidden state
            outputs = model.forward(**inputs, output_hidden_states=True, return_dict=True)
            full_hidden_state = outputs.hidden_states[0]
            # print(f"Full hidden state shape: {full_hidden_state.shape}")

    except Exception as e:
        print(f"Error during model execution: {e}")

    finally:
        # print("Removing the forward hook...")
        hook.remove()
        torch.cuda.empty_cache()

    return model_response, full_hidden_state, first_token_log_prob, last_token_log_prob


def main(args):
    model_size = args.model_size
    save_result = args.save_result
    intervention_factors = args.intervention_factor
    save_hidden_state = args.save_hidden_state
    global model, tokenizer, sae_model, scaling_factor, intervine_layer, intervine_latent_id,intervention_factor,variable
    variable = args.variable
    model, tokenizer, sae_model, scaling_factor, intervine_layer, intervine_latent_id = preapre_intervention(model_size)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(script_dir, '../output/'))

    max_trials = 500
    n_repeat = 5
    max_attempts = 5
    result_dict = {}

    for intervention_factor in tqdm(intervention_factors, desc="Intervention Factor"):
        print(f"Intervention Factor: {intervention_factor}")
        result_dict[intervention_factor] = {}
        for run in range(n_repeat):
            result_dict[intervention_factor][run] = {}
            # Define the system-level prompt (static for all trials)
            system_prompt = (
            "Welcome to the Alchemy Game! You start with four basic elements: water, fire, earth, and air.\n"
            "The objective of the game is to combine elements to create new ones. Each successful combination "
            "adds a new element to your inventory, which can be used for future combinations.\n"
            "Choose two elements to combine by writing them in the format 'element + element'. "
            "You can choose the same element twice or two different elements, and the order of the elements does not matter. Each combination produces deterministic."
            "Only output the combination, no other words. "
            "And you need try your best to create more elements. Let's get started!\n"
            )

            game = Game()
            for step in range(max_trials):
                # Update goals based on the current inventory
                game.check_goals()
                current_inventory = game.get_inventory_status()
                # Define the trial-specific prompt
                if not save_hidden_state:
                    trial_prompt = (
                    f"Current Inventory: {current_inventory}\n"
                    f"Past Attempts: {game.get_past_trials()}\n"
                    "Choose two elements to combine in the format 'element + element'. The output format should be 'element + element'. Only output the combination, no other words. Give me one combination every time: "
                    )
                else:
                    pass


                # Generate response from LLaMA
                valid_response = False
                for attempt in range(max_attempts):
                    # Global storage for tracking context and tokens
                    try:
                        llm_response, hidden_state, log_prob_element1, log_prob_element2 = run_intervention(model, tokenizer, sae_model, scaling_factor, intervine_layer, intervine_latent_id, system_prompt, trial_prompt, game.inventory.copy(),intervention_factor, save_result = save_result, save_hidden_state = save_hidden_state)
                        if not llm_response:
                            raise ValueError("Empty response from LLM")

                        print(f"LLaMA Response (Trial {step}, Attempt {attempt + 1}): {llm_response}")

                        # Parse response into elements
                        combination = llm_response.split("+")
                        if len(combination) != 2:
                            raise ValueError(f"Invalid format: Expected 'element + element', got '{llm_response}'")

                        # If we reach here, we have a valid response
                        valid_response = True
                        break

                    except Exception as e:
                        print(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                        continue

                if not valid_response:
                    print(f"Failed to get valid response after {max_attempts} attempts. Skipping trial {step}")
                    continue

                element1 = combination[0].strip().lower()
                element2 = combination[1].strip().lower()
            
                # Process the combination and update game state
                success, result, result_prompt = game.check_combination(element1, element2)

                prompt = system_prompt + "\n" + trial_prompt

                game.store_trial_info(run, step, (element1, element2), success, result, result_prompt, log_prob_element1, log_prob_element2, prompt)

                print(f"Trial {step} Completed: Success = {success}, Result = {result}, LogProbs = ({log_prob_element1}, {log_prob_element2})")

            combination_dict = {item['trial']: item for item in game.combination_storage}
            # remove each combineation_dict's sub_id and trial from the dictionary
            for item in combination_dict:
                del combination_dict[item]['id']
                del combination_dict[item]['trial']
            result_dict[intervention_factor][run]['results'] = combination_dict
            save_all_game_states(result_dict, save_path, model_size, max_trials, variable)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="8B", choices=["8B", "70B"])
    parser.add_argument("--intervention_factor", type=list, default=[0.5, 0.7])
    parser.add_argument("--variable", type=str, default="choices")
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--save_hidden_state", type=bool, default=False)
    args = parser.parse_args()
    main(args)
