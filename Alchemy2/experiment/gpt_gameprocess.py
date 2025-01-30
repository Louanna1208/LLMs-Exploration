import json
import openai
import os
import pandas as pd
import csv
import argparse
from openai import OpenAI
os.chdir(r':\GitHub\LLMs_game\Alchemy2')
#set the path
file_path = 'output/data/'
openai.api_key = 'api_key'
# Load Game Tree JSON for Element Mapping
with open(r'dataset/alchemy2Gametree.json', 'r') as f:
    game_tree = json.load(f)

# Create a lookup dictionary for element codes to names
element_code_to_name = {}
for code, element_data in game_tree.items():
    element_code_to_name[int(code)] = element_data["name"]

# Load Combination Table
combination_table = {}
with open(r'dataset/alchemy2CombinationTable.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        first = int(row['first'])
        second = int(row['second'])
        success = int(row['success'])
        result = int(row['result'])

        # Ensure each combination key maps to a list of results
        for key in [(first, second), (second, first)]:
            if key not in combination_table:
                combination_table[key] = []  # Initialize an empty list for each new key
            combination_table[key].append((success, result))  # Append each result to the list



class Game:
    def __init__(self, type):
        # Initialize inventory with the names of the first four elements
        self.inventory = {
            element_code_to_name[0],  # "water"
            element_code_to_name[1],  # "fire"
            element_code_to_name[2],  # "earth"
            element_code_to_name[3],  # "air"
        }
        # Provide 160 trials as initial trials
        self.combination_storage = []  # Stores all trial information
        self.inventory_size = len(self.inventory)
        # set the type of the game (inluding base, prompt_engineering, reasoning)
        self.type = type

    def check_goals(self):
        # Multiple-Stage Goals: Add elements to inventory based on inventory size
        # Check for size conditions
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

    def store_trial_info(self,run, step, combination, success, new_result, result_prompt,reason, top_logprob,config):
        # Store information about each trial
        if self.type == 'base' or self.type == 'prompt_engineering':
            trial_data = {
                'id': run,
                'trial': step,
                'first': combination[0],
                'second': combination[1],
                'success': success,
                'result': new_result if new_result else -1,
                'result_prompt': result_prompt,
                'inventory': list(self.inventory),
                'top_logprob': top_logprob,
                'config': config
            }

        elif self.type == 'reasoning':
            trial_data = {
                'id': run,
                'trial': step,
                'first': combination[0],
                'second': combination[1],
                'success': success,
                'result': new_result if new_result else -1,
                'result_prompt': result_prompt,
                'reason': reason,
                'inventory': list(self.inventory),
                'top_logprob': top_logprob,
                'config': config
            }

        self.combination_storage.append(trial_data)


    def get_past_trials(self):
        # Return a string representing past trials for prompting
        return " \n ".join([f"Trial {t['trial']}: {t['first']} + {t['second']} -> "
                           f"Success: {t['success']}, Result: {t['result']}, " 
                           f"{t['result_prompt']}"
                           for t in self.combination_storage])

    def get_inventory_status(self):
        # Return a string representing current inventory for prompting
        return ", ".join([f"{element}" for element in self.inventory])

    def check_combination(self, element1, element2):
        # Find the corresponding element codes using the name-to-code dictionary
        result_prompt = ""
        code1 = next((code for code, name in element_code_to_name.items() if name == element1), None)
        code2 = next((code for code, name in element_code_to_name.items() if name == element2), None)

        #print(f"Debug: code1 for {element1} = {code1}, code2 for {element2} = {code2}")

        # Check if the combination exists in the combination table (consider both possible orders)
        if code1 is not None and code2 is not None:
            # Retrieve all possible results for the combination regardless of order
            results = combination_table.get((code1, code2))

            #print(f"Debug: Retrieved results for combination ({element1}, {element2}) = {results}")

            if results:
                # Extract valid result codes
                possible_results = [result_code for success, result_code in results if success == 1]

                # If there are valid results, return the first new element not in inventory
                for result_code in possible_results:
                    result_name = element_code_to_name.get(result_code)
                    if result_name and result_name not in self.inventory:
                        # Add the new element to the inventory
                        self.inventory.add(result_name)
                        result_prompt = f"You successfully created a new element."
                        #print(f"Debug: Adding {result_name} to inventory")
                        return 1, result_name, result_prompt  # Return the new element

                # If all possible results are already in the inventory
                if possible_results:
                    last_result_name = element_code_to_name.get(possible_results[-1], None)
                    if last_result_name:
                        result_prompt = f"You already created this element before. Please select another element combination."
                        return 1, last_result_name, result_prompt

                # If results contain only unsuccessful combinations
                if all(success == 0 for success, result_code in results):
                    result_prompt = "Combining these two elements failed. Please select another element combination."
                    return 0, None, result_prompt

        print("Debug: Combination not found or unsuccessful.")
        result_prompt = "Combining these two elements failed. Please select another element combination."
        return 0, None, result_prompt  # Combination not found or unsuccessful


    def process_and_print_trials(self):
        # Process and print each trial information
        for trial in self.combination_storage:
            print(f"Run ID: {trial['id']}, Trial: {trial['trial']}, "
                  f"Combination: {trial['first']} + {trial['second']}, "
                  f"Success: {trial['success']}, Result: {trial['result']}, "
                  f"Inventory: {trial['inventory']}")

def call_gpt(prompt, config, type):
    try:
        # Provide game introduction (system prompt) for different types of game
        if type == 'base':
            initial_prompt = (
            "Welcome to the Alchemy Game! You start with four basic elements: water, fire, earth, and air.\n"
            "The objective of the game is to combine elements to create new ones. Each successful combination "
            "adds a new element to your inventory, which can be used for future combinations.\n"
            "Choose two elements to combine by writing them in the format 'element + element'. "
            "You can choose the same element twice or two different elements, and the order of the elements does not matter. Each combination produces deterministic."
            "Only output the combination, no other words. "
            "And you need try your best to create more elements. Let's get started!\n"
            )
        elif type == 'prompt_engineering':
            initial_prompt = ( 
            "Welcome to the Alchemy Game! You start with four basic elements: water, fire, earth, and air.\n"
            "The objective of the game is to combine elements to create new ones. Each successful combination "
            "adds a new element to your inventory, which can be used for future combinations.\n"
            "You can choose the same element twice or two different elements, and the order of the elements does not matter. "
            "Each combination produces deterministic results, meaning there is only one fixed outcome for each combination. Results cannot change upon repetition. Do not hypothesize new outcomes for previously combinations."
            "Step 1: Analyze past attempts to identify patterns (e.g., which elements combine well).\n"
            "Step 2: Choose a new combination based on patterns and unused pairs.\n"
            "And you need try your best to create more elements. Let's get started!\n"
            )
        elif type == 'reasoning':
            initial_prompt = (
            "Welcome to the Alchemy Game! You start with four basic elements: water, fire, earth, and air.\n"
            "The objective of the game is to combine elements to create new ones. Each successful combination "
            "adds a new element to your inventory, which can be used for future combinations.\n"
            "You can choose the same element twice or two different elements, and the order of the elements does not matter. "
            "Each combination produces deterministic results, meaning there is only one fixed outcome for each combination. Results cannot change upon repetition. Do not hypothesize new outcomes for previously combinations."
            "And you need try your best to create more elements. Let's get started!\n"
            )
        if config['model'] == 'gpt-4o-2024-08-06':
            response = openai.chat.completions.create(
                model = config['model'],
                messages = [{"role": "system", "content": initial_prompt},{"role": "user", "content": prompt}],
                max_tokens = 50,
                temperature = config['temperature'],  # Adjust for exploration vs. exploitation
                seed = config['seed'],
                logprobs = config['logprobs'],
                top_logprobs = config['top_logprobs']
             )
            #return the content, logprobs and top_logprobs
            return response.choices[0].message.content, response.choices[0].logprobs.content[0].top_logprobs
        elif config['model'] == 'o1-2024-12-17':
            response = openai.chat.completions.create(
                model = config['model'],
                messages = [{"role": "system", "content": initial_prompt},{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content, None
        elif config['model'] == 'deepseek-reasoner':
            client = OpenAI(api_key="api_key", base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model = config['model'],
                messages = [{"role": "system", "content": initial_prompt},{"role": "user", "content": prompt}],
                max_tokens = 500,
                temperature = config['temperature'],  # Adjust for exploration vs. exploitation
                stream = False
            )
            return response.choices[0].message.content, None, response.choices[0].message.reasoning_content
    
    except Exception as e:
        print(f"Error calling {config['model']} API: {e}")
        return None

# save_all_game_states
def save_all_game_states(result_dict, file_path, config, type):
    # Save all results into a single JSON file after all runs complete
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directories if they don't exist

    # Save the complete result dictionary to one JSON file
    with open(file_path + f'{type}_{config["model"]}_{config["max_trials"]}_results.json', 'w') as file:
        json.dump(result_dict, file, indent=4)
    print(f"All game runs saved to {file_path}{type}_{config['model']}_{config['max_trials']}_results.json")

def play_game(result_dict,max_trials, run_id, config, type):
    game = Game(type)
    for step in range(max_trials):
        # Update the goals based on inventory size
        game.check_goals()

        # Create a prompt with the current inventory and past trials
        if type == 'base':
            prompt = (
                f"Current Inventory: {game.get_inventory_status()}\n"
                f"Past Attempts: {game.get_past_trials()}\n"
                "Choose two elements to combine in the format 'element + element'. The output format should be 'element + element'. Only output the combination, no other words. Give me one combination every time: "
            )

        elif type == 'prompt_engineering':
            prompt = (
                f"Current Inventory: {game.get_inventory_status()}\n"
                f"Past Attempts: {game.get_past_trials()}\n"
                "Before finalizing, check:\n"
                "- Has this combination been tried before? \n"
                "- If yes, choose a new one. \n"
                "- If no, proceed. \n"
                "The output format should be 'element1 + element2'. Only output the element combination, no other words. Give me one combination every time."
            )
        elif type == 'reasoning':
            prompt = (
                f"Current Inventory: {game.get_inventory_status()}\n"
                f"Past Attempts: {game.get_past_trials()}\n"
                "The output format should be 'reasoning process:XXX. I will choose element + element'. Only output your reasoning process and the element you choose, no other words. Give me one combination every time."
            )

        print(prompt)

        # Call GPT-4 to get the combination decision
        llm_response = call_gpt(prompt, config, type)
        if not llm_response:
            print(f"Skipping due to {config['model']} API error")
            continue
        print(f"{config['model']} Response: {llm_response[0]}")

        # Parse GPT-4 response to get the combination, logprobs and top_logprobs
        if type == 'base' or type == 'prompt_engineering':
            combination = llm_response[0].split("+")
            if len(combination) != 2:
                print(f"Invalid response from LLM: {llm_response}")
                continue
        elif type == 'reasoning':
            # the combination is the last 3 tokens of the response
            if "I will choose" in llm_response[0]:
                reasoning_part, combination_part = llm_response[0].split("I will choose", 1)
                combination = combination_part.strip(".")
                combination = combination.split("+")
                if len(combination) != 2:
                    print(f"Invalid response from LLM: {llm_response}")
                    continue
            elif "+" in llm_response[0]:
                # Split the response into sentences and get the last sentence
                sentences = llm_response[0].split(".")
                last_sentence = sentences[-1].strip()

                # Find and extract the combination
                if "+" in last_sentence:
                    combination = last_sentence.strip()
                    combination = combination.split("+")
                    if len(combination) != 2:
                        print(f"Invalid response from LLM: {llm_response}")
                        continue
                else:
                    print("No valid combination found in the last sentence.")
                    continue
            else:
                print("Invalid response format.")
                continue

        element1 = combination[0].strip().lower()
        element2 = combination[1].strip().lower()

        # Check combination success and update inventory
        success, result, result_prompt = game.check_combination(element1, element2)

        # post processing the top_logprobs
        if llm_response[1] is None:
            logprob_dict = None
        else:
            logprob_dict = {item.token: item.logprob for item in llm_response[1]}
        #save the game state
        if run_id not in result_dict[config['model']]:
            result_dict[config['model']][run_id] = {}
        # Store the trial information including logprobs and top_logprobs
        if config['model'] == 'deepseek-reasoner':
            game.store_trial_info(run_id, step, (element1, element2), success, result, result_prompt, llm_response[2],logprob_dict, config.copy())
        else:
            game.store_trial_info(run_id, step, (element1, element2), success, result, result_prompt, llm_response[0],logprob_dict, config.copy())

    combination_dict = {item['trial']: item for item in game.combination_storage}
    # remove each combineation_dict's sub_id and trial from the dictionary
    for item in combination_dict:
        del combination_dict[item]['id']
        del combination_dict[item]['trial']

    result_dict[config['model']][run_id] = combination_dict

def main():
    parser = argparse.ArgumentParser(description='Run Alchemy2 experiments with GPT')
    # o1 can't be used for reasoning, it will has an error( Error code: 400. 'Invalid prompt: your prompt was flagged as potentially violating our usage policy.)
    # deepseek-model choose the base type, it can output the reasoning process
    parser.add_argument('--type', type=str, required=True,
                      choices=['base', 'prompt_engineering', 'reasoning'],
                      help='Type of the game(base, prompt_engineering, reasoning)')
    parser.add_argument('--model', type=str, required=True,
                      choices=['gpt-4o-2024-08-06', 'o1-2024-12-17', 'deepseek-reasoner'],
                      help='Model to use(gpt-4o-2024-08-06, o1-2024-12-17, deepseek-reasoner)')
    parser.add_argument('--max_trials', type=int, required=True,
                      help='Maximum number of trials')
    parser.add_argument('--n_repeats', type=int, required=True,
                      help='Number of repeats for each temperature and model')
    
    args = parser.parse_args()
    run_id = 0
    result_dict = {args.model: {}}
    
    # set a config for the whole game
    if args.model == 'gpt-4o-2024-08-06':
        config = {
            'max_trials': args.max_trials,
            'temperature': 1,
            'seed': 1,
            'logprobs': True,
            'top_logprobs': 20,
            'model': args.model
        }
        for config['temperature'] in [0, 0.3, 0.7, 1]:
            for _ in range(args.n_repeats):
                run_id += 1
                config['seed'] = run_id
                play_game(result_dict, config['max_trials'], run_id, config, args.type)
        save_all_game_states(result_dict,file_path,config, args.type)
    elif args.model == 'o1-2024-12-17':
        config = {
            'max_trials': args.max_trials,
            'model': args.model
        }
        for _ in range(args.n_repeats):
            run_id += 1
            play_game(result_dict, config['max_trials'], run_id, config, args.type)
        # save the result_dict after all runs are completed
        save_all_game_states(result_dict,file_path,config, args.type)
    elif args.model == 'deepseek-reasoner':
        config = {
            'max_trials': args.max_trials,
            'temperature': 1,
            'model': args.model
        }
        for _ in range(args.n_repeats):
            run_id += 1
            play_game(result_dict, config['max_trials'], run_id, config, args.type)
        save_all_game_states(result_dict,file_path,config, args.type)

if __name__ == "__main__":
    main()