# Encapsulate this code into a callable function data_preprocessing()
import pandas as pd
import json
import os
os.chdir('/Github/LLMs_game/Alchemy2')
from ast import literal_eval
# Step 1: Create a lookup dictionary for element codes to names
element_code_to_name = {}
with open(r'dataset/alchemy2Gametree.json', 'r') as f:
    game_tree = json.load(f)
for code, element_data in game_tree.items():
    element_code_to_name[int(code)] = element_data["name"]
# Create a lookup dictionary for element names to codes
element_name_to_code = {v: k for k, v in element_code_to_name.items()}

# Step 2: Load the LLM_player_data json file
def LLM_data_preprocessing(file_path,model_name):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Adjust data extraction to focus specifically on relevant keys
    if model_name == 'gpt-4o':
        records = []
        # Loop over models, players, and temperature settings
        for model, players in data.items():
            for id, trials in players.items():
                if isinstance(trials, dict):
                    for trial, trial_data in trials.items():
                        # Only consider numeric trial numbers to avoid non-trial entries
                        if trial.isdigit():
                            # Ensure all necessary keys are present in trial_data
                            if all(key in trial_data for key in ["first", "second", "success", "result", "inventory","config"]):
                                # Access seed and temperature from the trial's config
                                seed = trial_data["config"].get("seed")
                                temperature = trial_data["config"].get("temperature")
                                record = {
                                "model": model,
                                "id": int(id),  # Convert player to integer for mapping purposes
                                "trial": int(trial),
                                "first": trial_data["first"],
                                "second": trial_data["second"],
                                "success": trial_data["success"],
                                "results": trial_data["result"],
                                "inventory_names": trial_data["inventory"],
                                "seed": seed,
                                "temperature": temperature,
                                "top_logprob": None,
                            }
                            records.append(record)
    elif model_name == 'Llama3':
        records_by_player = {}
        for model, players in data.items():
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
                               # "prompt": trial_data["prompt"],
                            }   
                            player_records.append(record)
                    records_by_player[player_id] = player_records
                # If you want one flat list of all players: 
        records = []
        for player_id, player_records in records_by_player.items():
            records.extend(player_records)

    # Create a DataFrame from the extracted records
    LLM_player_data = pd.DataFrame(records)
    # Filter rows where both 'first' and 'second' are in the inventory_names list(There need all trials so we don't filter)
    #LLM_player_data = LLM_player_data[
    #LLM_player_data.apply(lambda row: row['first'] in row['inventory_names'] and row['second'] in row['inventory_names'], axis=1)
    #]
    LLM_with_names = LLM_player_data.copy()
    # make the result =-1 is the name "failed"
    LLM_with_names['results'] = LLM_with_names['results'].apply(
        lambda x: 'failed' if x == -1 else x
    )
    # add the length of inventory_names to LLM_with_names
    LLM_with_names['inventory'] = LLM_with_names['inventory_names'].apply(len)
    LLM_with_codes = LLM_player_data.copy()
    LLM_with_codes['first'] = LLM_with_codes['first'].map(element_name_to_code)
    LLM_with_codes['second'] = LLM_with_codes['second'].map(element_name_to_code)
    # make the result into codes,result=-1 is -1
    LLM_with_codes['results'] = LLM_with_codes['results'].apply(
        lambda x: element_name_to_code.get(x, x)
        )
    # make the inventory_names into codes
    LLM_with_codes['inventory_names'] = LLM_with_codes['inventory_names'].apply(lambda inventory: [element_name_to_code[element] for element in inventory])
    if model_name == 'Llama3':
        LLM_with_codes['inventory_elements'] = LLM_with_codes['inventory_elements'].apply(lambda inventory: [element_name_to_code[element] for element in inventory])
    # add the length of inventory_names to LLM_with_codes
    LLM_with_codes['inventory'] = LLM_with_codes['inventory_names'].apply(len)

    return LLM_with_names, LLM_with_codes

#Llama3 data, this dataset is preparing for the SAE
Llama3_70B_LLM_data_path = 'output/data/Llama3_70B_500_results.json'
Llama3_70B_LLM_with_names, Llama3_70B_LLM_with_codes = LLM_data_preprocessing(Llama3_70B_LLM_data_path,'Llama3')
save_path_names = 'output/data/Llama3_70B_LLM_results(full_names).csv'
Llama3_70B_LLM_with_names.to_csv(save_path_names, index=False)
save_path_codes = 'output/data/Llama3_70B_LLM_results(full_codes).csv'
Llama3_70B_LLM_with_codes.to_csv(save_path_codes, index=False)

Llama3_8B_LLM_data_path = 'output/data/Llama3_8B_500_results.json'
Llama3_8B_LLM_with_names, Llama3_8B_LLM_with_codes = LLM_data_preprocessing(Llama3_8B_LLM_data_path,'Llama3')
save_path_names = 'output/data/Llama3_8B_LLM_results(full_names).csv'
Llama3_8B_LLM_with_names.to_csv(save_path_names, index=False)
save_path_codes = 'output/data/Llama3_8B_LLM_results(full_codes).csv'
Llama3_8B_LLM_with_codes.to_csv(save_path_codes, index=False)

Base_LLM_data_path = 'output/data/base_gpt-4o-2024-08-06_500_results.json'
Base_LLM_with_names, Base_LLM_with_codes = LLM_data_preprocessing(Base_LLM_data_path,'gpt-4o')
save_path_names = 'output/data/base_LLM_results(full_names).csv'
Base_LLM_with_names.to_csv(save_path_names, index=False)
save_path_codes = 'output/data/base_LLM_results(full_codes).csv'
Base_LLM_with_codes.to_csv(save_path_codes, index=False)

# before the SAE, we can check the elements in the dataset
#elements = []
#data = pd.read_csv("D:/Github/LLMs_game/Alchemy2/output/data/Llama3_8B_LLM_results(full_codes).csv")
#print(f"len: {len(data['inventory_elements'])}")
#for inventory_elements in data['inventory_elements']:
#    inventory_elements = literal_eval(inventory_elements)
#    for element in inventory_elements:
#        elements.append(element)
#print(f"len: {len(elements)}")



