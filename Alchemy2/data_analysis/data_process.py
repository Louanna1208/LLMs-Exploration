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
    records = []
    if model_name == 'gpt-4o':
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
        for model, players in data.items():
            for id, player_data in players.items():
                if isinstance(player_data, dict) and "config" in player_data:  # Ensure 'config' key exists
                    config = player_data["config"]
                    for trial, trial_data in player_data["results"].items():
                        if trial.isdigit():
                            if all(key in trial_data for key in ["first", "second", "success", "result", "inventory","log_prob_first","log_prob_second"]):
                                record = {
                                    "model": model,
                                    "id": int(id),  # Convert player to integer for mapping purposes
                                    "trial": int(trial),
                                    "first": trial_data.get("first",None),# some trials may not have first
                                    "second": trial_data.get("second",None),
                                    "success": trial_data["success"],
                                    "results": trial_data["result"],
                                    "inventory_names": trial_data["inventory"],
                                    "seed": None,
                                    "temperature": config["temperature"],
                                    "top_logprob": {
                                        "log_prob_first":trial_data.get("log_prob_first"),
                                        "log_prob_second":trial_data.get("log_prob_second")
                                        },
                                }   
                            records.append(record)
    elif model_name == 'LLaMA3.1_intervention':
        for intervention, players in data.items():
            for id, player_data in players.items():
                if isinstance(player_data, dict):  
                    for trial, trial_data in player_data["results"].items():
                        if trial.isdigit():
                            if all(key in trial_data for key in ["first", "second", "success", "result", "inventory","log_prob_first","log_prob_second"]):
                                record = {
                                    "model": intervention,
                                    "id": int(id),  # Convert player to integer for mapping purposes
                                    "trial": int(trial),
                                    "first": trial_data.get("first",None),# some trials may not have first
                                    "second": trial_data.get("second",None),
                                    "success": trial_data["success"],
                                    "results": trial_data["result"],
                                    "inventory_names": trial_data["inventory"],
                                    "seed": None,
                                    # make all temperature 1
                                    "temperature": 1,
                                    "top_logprob": {
                                        "log_prob_first":trial_data.get("log_prob_first"),
                                        "log_prob_second":trial_data.get("log_prob_second")
                                        },
                                }   
                            records.append(record)

    # Create a DataFrame from the extracted records
    LLM_player_data = pd.DataFrame(records)

    # Reorganize DataFrame columns for clarity
    LLM_player_data = LLM_player_data[["model", "seed", "id", "temperature", "trial", "first", "second", "success", "results", "inventory_names", "top_logprob"]]
    # Filter rows where both 'first' and 'second' are in the inventory_names list
    LLM_player_data = LLM_player_data[
    LLM_player_data.apply(lambda row: row['first'] in row['inventory_names'] and row['second'] in row['inventory_names'], axis=1)
]
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
    # add the length of inventory_names to LLM_with_codes
    LLM_with_codes['inventory'] = LLM_with_codes['inventory_names'].apply(len)

    return LLM_with_names, LLM_with_codes


def Human_data_preprocessing(file_path):
    # Step 2: Load Human Data
    human_data = pd.read_csv(file_path) 

# (3) Human Players Result with Element Codes
    Human_with_codes = human_data.copy()

# (4) Human Players Result with Element Names
    Human_with_names = Human_with_codes.copy()
    Human_with_names['first'] = Human_with_names['first'].map(element_code_to_name)
    Human_with_names['second'] = Human_with_names['second'].map(element_code_to_name)
    # convert element in Human_with_names['results'] from list to int
    Human_with_names['results'] = Human_with_names['results'].apply(literal_eval)
    # make result=-1 is the name "failed"
    # make the result element names to codes, and the result is a list of codes
    Human_with_names['results'] = Human_with_names['results'].apply(
        lambda x: 'failed' if x == -1 
        else [element_code_to_name.get(i, i) for i in x] if isinstance(x, list) 
        else element_code_to_name.get(x, x)
        )
    return Human_with_names, Human_with_codes
