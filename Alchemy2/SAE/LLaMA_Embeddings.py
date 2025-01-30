import os
import json
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import cuda, bfloat16
import transformers
from ast import literal_eval
import re

#os.environ['TRANSFORMERS_CACHE'] = "E:/Huggingface/model_cache"

# Load LLaMA model and tokenizer
def model_loading(model_id):
    API_key = 'api_key'


    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # begin initializing HF items, need auth token for these
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=API_key
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        # device_map={'': 0},
        device_map='auto',
        token=API_key,
        torch_dtype = 'auto'
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=API_key
    )
    return model, tokenizer

# load trial_prompt and inventory_elements
def load_trial_prompt_and_inventory_elements(model_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = os.path.abspath('C:/Users/Louanna/Documents/GitHub/LLMs_game/Alchemy2/SAE')
    with open(os.path.join(script_dir, f'../output/data/Llama3_{model_size}_500_results.json'), 'r') as f:
        Llama3_results = json.load(f)
        records_by_player = {}
    for model, players in Llama3_results.items():
        # only consider the last 5 players who's temperature is 1.0
        players = list(players.items())[-5:]
        for id, player_data in players:
            if isinstance(player_data, dict): 
                player_id = int(id)
                player_records = []
                for trial, trial_data in player_data["results"].items():
                    trial_idx = int(trial)
                    if trial_idx == 0:
                        # Initial elements
                        inventory_elements = ["water", "air", "earth", "fire"]
                    else:
                        # For non-zero trials, inventory_elements from the previous trial's inventory_names
                        prev_inventory_names = player_records[trial_idx - 1]["inventory_names"]
                        inventory_elements = prev_inventory_names
                    if trial.isdigit():
                        record = {
                            "model": model,
                            "id": int(id),  # Convert player to integer for mapping purposes
                            "trial": int(trial),
                            "first": trial_data.get("first",None),# some trials may not have first
                            "second": trial_data.get("second",None),
                            "success": trial_data["success"],
                            "results": trial_data["result"],
                            "inventory_names": trial_data["inventory"] if isinstance(trial_data["inventory"], list) else literal_eval(trial_data["inventory"]),
                            "inventory_elements": inventory_elements if isinstance(inventory_elements, list) else literal_eval(inventory_elements),
                            "prompt": trial_data["prompt"]
                        }   
                        player_records.append(record)
                records_by_player[player_id] = player_records
                # If you want one flat list of all players:
    records = []
    for player_id, player_records in records_by_player.items():
        records.extend(player_records)

    elements = []
    inventory_elements = (record['inventory_elements'] for record in records)
    for inventory_element in inventory_elements:
        for element in inventory_element:
            elements.append(element)

    print(f"Total elements: {len(elements)}")
    return records



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


# Save batch embeddings to file
def save_batch(embedding_list, model_size, batch_count):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = f'../embedding_result/dataset/Batch_embedding'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(script_dir, f'{folder_name}/LLaMA3_{model_size}_embedding_batch{batch_count}.npy')
    np.save(filename, np.array(embedding_list))
    print(f"Batch {batch_count} saved to {filename}")

def generate_inventory_embeddings(trial_prompts, inventory_elements_list, records, model_size):
    """
    Generate and save embeddings for inventory elements in batches.
    """
    model, tokenizer = model_loading(f'meta-llama/Meta-Llama-3.1-{model_size}-Instruct')
    results = []
    embedding_list = []
    batch_count = 0
    checkpoint_interval = 100  # Number of rows after which to save a batch
    num_element = 0
    for i, (trial_prompt, inventory_elements, record) in tqdm(enumerate(zip(trial_prompts, inventory_elements_list, records)), total=len(trial_prompts)):
        # Locate absolute positions of elements in the full prompt

        element_positions = locate_token_positions_full_prompt(trial_prompt, inventory_elements, tokenizer)

        # Tokenize the full prompt and get embeddings
        inputs = tokenizer(trial_prompt, return_tensors="pt", truncation=True)
        inputs = inputs.to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # List of tensors (num_layers, batch_size, seq_len, embedding_dim)

        # Extract embeddings for inventory elements
        element_embeddings = []
        
        for element, positions in element_positions.items():
            # Average embeddings across all layers for each position
            embeddings =  torch.stack([
                torch.mean(layer[:, positions, :], dim=1)  # Average across the specified positions
                for layer in hidden_states
            ])
            embeddings = embeddings.squeeze(1)
            #convert the tensor to float
            embeddings = embeddings.float()
            # Final shape is (num_layers, embedding_dim)
            embeddings = embeddings.detach().cpu().numpy()
            element_embeddings.append(embeddings)
            
        
        element_embeddings = np.array(element_embeddings)
        num_element += element_embeddings.shape[0]
        print(f"num_element: {num_element}")
        print(f"element_embeddings shape: {element_embeddings.shape}")
        # detach and clear the GPU memory
        del hidden_states, outputs
        torch.cuda.empty_cache()

        embedding_list.append(element_embeddings)
        # Save embeddings in batches
        if len(embedding_list) >= checkpoint_interval:
            batch_count += 1
            embedding_list = np.concatenate(embedding_list, axis=0)
            save_batch(embedding_list, model_size, batch_count)
            embedding_list = []

    # Save remaining embeddings
    if embedding_list:
        batch_count += 1
        save_batch(embedding_list, model_size, batch_count)
    print(f"num_element: {num_element}")
    # clear the GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def extract_batch_number(filename):
    # Extract the number between 'batch' and '.pt'
    match = re.search(r'batch(\d+)\.npy$', filename)
    return int(match.group(1)) if match else 0
    
# Transform batch embeddings into layer-specific embeddings
def transform_batch_to_layer(model_size, overwrite = False, chunk_size=1):
    """
    Transform batch embeddings to layer embeddings with improved memory efficiency.
    Uses memory mapping and processes files in chunks to reduce memory usage.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, '..', 'embedding_result', 'dataset', 'Batch_embedding')
    layer_folder_name = os.path.join(script_dir, '..','embedding_result', 'dataset', 'Layer_embedding')
    # Ensure the output directory exists
    if not os.path.exists(layer_folder_name):
        os.makedirs(layer_folder_name)


    # Get all npy files
    npy_files = sorted([f for f in os.listdir(folder_name) if f.endswith('.npy')], 
                 key=extract_batch_number)
    
    # Use memory mapping to get shape information
    sample_mmap = np.load(os.path.join(folder_name, npy_files[0]), mmap_mode='r')
    num_layers = sample_mmap.shape[1]
    embedding_dim = sample_mmap.shape[2]
    del sample_mmap

    # check how many layers embeddings have existed
    existing_layers = [int(f.split('layer')[1].split('.npy')[0]) for f in os.listdir(layer_folder_name) if f.endswith('.npy')]
    # start from the layer that has not been processed but exclude last existing layer since it's possibly incomplete
    if overwrite == False:
        start_layer = max(existing_layers) if existing_layers else 0
    else:
        start_layer = 0

    
    # Process each layer
    for layer in tqdm(range(start_layer, num_layers), desc='Processing layers'):
        num_element = 0
        output_file = os.path.join(layer_folder_name, f'LLaMA3_{model_size}_embedding_layer{layer}.npy')
        
        # Handle long paths or file issues
        if len(output_file) > 240:
            raise OSError(f"Path too long: {output_file}")
        if os.path.exists(output_file):
            os.remove(output_file)  # Remove existing file if needed
        # Process files in chunks
        for chunk_start in range(0, len(npy_files), chunk_size):
            chunk_files = npy_files[chunk_start:chunk_start + chunk_size]
            chunk_embeddings = []
            
            # Process each file in the chunk
            for file in chunk_files:
                # Use memory mapping to load the file
                mmap_data = np.load(os.path.join(folder_name, file), mmap_mode='r')
                # Extract only the needed layer and copy to memory
                layer_data = mmap_data[:, layer, :].copy()
                chunk_embeddings.append(layer_data)
                num_element += layer_data.shape[0]
                print(f"num_element: {num_element}")
                del mmap_data
                
            # Concatenate chunk results
            chunk_result = np.concatenate(chunk_embeddings, axis=0)
            # Save or append results
            if chunk_start == 0:
                # For first chunk, create new file
                np.save(output_file, chunk_result)
            else:
                # For subsequent chunks, append to existing file
                existing_data = np.load(output_file)
                combined = np.concatenate([existing_data, chunk_result])
                np.save(output_file, combined)
                del existing_data

            # Clean up
            del chunk_embeddings
            del chunk_result
            gc.collect()  # Force garbage collection
        print(f"num_element: {num_element}")



# Main execution
def main():
    parser = argparse.ArgumentParser(description="Generate inventory embeddings")
    parser.add_argument("--model_size", type=str, default="70b", help="Specify model size, e.g., '8b', '70b'")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite previous embeddings.")
    args = parser.parse_args()

    model_size = args.model_size
    overwrite = args.overwrite
    records = load_trial_prompt_and_inventory_elements(model_size)
    trial_prompts = [record['prompt'] for record in records]
    print(f"trial_prompts: {len(trial_prompts)}")  
    inventory_elements = [record['inventory_elements'] for record in records]
    print(f"inventory_elements: {len(inventory_elements)}")
    # generate_inventory_embeddings(trial_prompts, inventory_elements, records, model_size)
    torch.cuda.empty_cache()
    gc.collect()
    transform_batch_to_layer(model_size, overwrite)

if __name__ == "__main__":
    main()

