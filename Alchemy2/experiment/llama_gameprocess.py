from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pickle
import json
import os
import transformers
#os.chdir(r'\Github\LLMs_game\Alchemy2')
import csv
from torch import cuda, bfloat16
from tqdm import tqdm
import argparse
#file_path = 'output/'

# os.environ['TRANSFORMERS_CACHE'] = "/HuggingFace/model_cache"

# Save and load pickle files
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

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
# Define the Game class
class Game:
    def __init__(self):
        self.inventory = {
            element_code_to_name[0],  # "water"
            element_code_to_name[1],  # "fire"
            element_code_to_name[2],  # "earth"
            element_code_to_name[3],  # "air"
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
            element_name = element_code_to_name.get(element_code)
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
        code1 = next((code for code, name in element_code_to_name.items() if name == element1), None)
        code2 = next((code for code, name in element_code_to_name.items() if name == element2), None)

        if code1 is not None and code2 is not None:
            results = combination_table.get((code1, code2))

            if results:
                possible_results = [result_code for success, result_code in results if success == 1]

                for result_code in possible_results:
                    result_name = element_code_to_name.get(result_code)
                    if result_name and result_name not in self.inventory:
                        self.inventory.add(result_name)
                        result_prompt = f"You successfully created a new element."
                        return 1, result_name, result_prompt

                if possible_results:
                    last_result_name = element_code_to_name.get(possible_results[-1], None)
                    if last_result_name:
                        result_prompt = f"You already created this element before. Please select another element combination."
                        return 1, last_result_name, result_prompt

                if all(success == 0 for success, result_code in results):
                    result_prompt = "Combining these two elements failed. Please select another element combination."
                    return 0, None, result_prompt

        result_prompt = "Combining these two elements failed. Please select another element combination."
        return 0, None, result_prompt

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
        bnb_4bit_compute_dtype=bfloat16
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

# Generate response from LLaMA
def call_llama(system_prompt, trial_prompt,config, tokenizer, model, device):
    try:
        # Tokenize and process input
        prompt = system_prompt + "\n" + trial_prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_tokens'],
            temperature=max(config['temperature'], 1e-10),
            do_sample=True,  # Sample from the distribution
            return_dict_in_generate=True, 
            output_scores=True
        )
        # Decode and return the response
        generated_tokens = outputs.sequences[:, -config['max_tokens']:] 
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        # get the first and the last token's log likelihood in generated_tokens
        first_token_log_prob = transition_scores[0][0].item()
        last_token_log_prob = transition_scores[-1][0].item()
        response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return response, first_token_log_prob, last_token_log_prob
    except Exception as e:
        print(f"Error generating response with LLaMA: {e}")
        return None


# save_all_game_states
def save_all_game_states(result_dict, save_path, config):
    # Create directories if they don't exist
    file_name = os.path.join(save_path, f'data/{config["model"]}/{config["max_trials"]}_results.json')
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))  # Create directories if they don't exist

    # Save the complete result dictionary to one JSON file
    with open(file_name, 'w') as file:
        json.dump(result_dict, file, indent=4)
    print(f"All game runs saved to {save_path}data/{config['model']}/{config['max_trials']}_results.json")

# play_game_with_llama
def play_game_with_llama(run_id, result_dict, config):
    """
    Main game loop using LLaMA for generating responses and computing log-probabilities.
    """

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

    for step in range(config['max_trials']):
        # Update goals based on the current inventory
        game.check_goals()

        # Define the trial-specific prompt
        trial_prompt = (
            f"Current Inventory: {game.get_inventory_status()}\n"
            f"Past Attempts: {game.get_past_trials()}\n"
            "Choose two elements to combine in the format 'element + element'. The output format should be 'element + element'. Only output the combination, no other words. Give me one combination every time: "
        )


        # Generate response from LLaMA
        valid_response = False
        for attempt in range(config['max_attempts']):
            try:
                llm_response, log_prob_element1, log_prob_element2 = call_llama(system_prompt, trial_prompt, config, tokenizer, llama_model, device)
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
                print(f"Attempt {attempt + 1}/{config['max_attempts']} failed: {str(e)}")
                continue

        if not valid_response:
            print(f"Failed to get valid response after {config['max_attempts']} attempts. Skipping trial {step}")
            continue

        element1 = combination[0].strip().lower()
        element2 = combination[1].strip().lower()
        
        # Process the combination and update game state
        success, result, result_prompt = game.check_combination(element1, element2)

        prompt = system_prompt + "\n" + trial_prompt

        game.store_trial_info(run_id, step, (element1, element2), success, result, result_prompt, log_prob_element1, log_prob_element2, prompt)

        print(f"Trial {step} Completed: Success = {success}, Result = {result}, LogProbs = ({log_prob_element1}, {log_prob_element2})")

    combination_dict = {item['trial']: item for item in game.combination_storage}
    # remove each combineation_dict's sub_id and trial from the dictionary
    for item in combination_dict:
        del combination_dict[item]['id']
        del combination_dict[item]['trial']

    result_dict[config['model']][run_id]['results'] = combination_dict

def run_experiment(model_name):
    # save the result_dict after all runs are completed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(script_dir, '../output/'))
    # Game Configuration
    config = {
    'max_trials': 500,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_tokens': 3,
    # model name include meta-llama/llama-3-8b-hf, meta-llama/llama-2-13b-hf, meta-llama/llama-3-70b-hf
    'model': model_name,
    'max_attempts': 5
    }
    n_repeats = 5 #repeat the game for each temperature and model
    run_id = 0
    global tokenizer, llama_model, device, element_code_to_name, combination_table, result_dict
    result_dict = {model_name: {}}
    element_code_to_name, combination_table = load_game_tree(script_dir)
    llama_model, tokenizer = model_loading(config['model'])
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    for config['temperature'] in [0, 0.3, 0.7, 1]:
        print(f'Running {config["model"]} with temperature: {config["temperature"]}')
        for _ in range(n_repeats):
            run_id += 1
            if run_id not in result_dict[config['model']]:
                result_dict[config['model']][run_id] = {}
                result_dict[config['model']][run_id]['config'] = config.copy()
            play_game_with_llama(run_id, result_dict, config)

    save_all_game_states(result_dict,save_path,config)

def main():
    parser = argparse.ArgumentParser(description='Run Alchemy2 experiments with LLaMA')
    parser.add_argument('--model', type=str, required=True,
                      choices=['meta-llama/Meta-Llama-3.1-8B-Instruct',
                               'meta-llama/Meta-Llama-3.1-70B-Instruct'],
                      help='Model to use')
    
    args = parser.parse_args()
    
    # Run experiment
    run_experiment(args.model)

if __name__ == "__main__":
    main()