import pandas as pd
import os
import numpy as np
import math
from tqdm import tqdm
import json
import ast
from sklearn.preprocessing import StandardScaler
from ast import literal_eval

os.chdir('/Github/LLMs_game/Alchemy2')

class element_value:
    def __init__(self):
        self.n_elements = 720
        self.values_elements_cbu = np.zeros(self.n_elements)
        self.element_count_cbu = np.ones(self.n_elements)
        self.values_elements_cbv = np.zeros(self.n_elements)
        self.values_elements_rec = np.zeros(self.n_elements)
        self.element_lastused_rec = np.zeros(self.n_elements)
        self.total_count_cbu = 1
        self.total_count_rec = 1

    def reset_model(self, model_type):
        if model_type == 'cbu':
            self.values_elements_cbu = np.zeros(self.n_elements)
            self.element_count_cbu = np.ones(self.n_elements)
            self.total_count_cbu = 1
        elif model_type == 'cbv':
            self.values_elements_cbv = np.zeros(self.n_elements)
        elif model_type == 'rec':
            self.values_elements_rec = np.zeros(self.n_elements)
            self.element_lastused_rec = np.zeros(self.n_elements)
            self.total_count_rec = 1

    def cbu_model(self, chosen_combination, update):
        if update:
            self.total_count_cbu += 1
            self.element_count_cbu[chosen_combination[0]] += 1
            self.element_count_cbu[chosen_combination[1]] += 1
            for i in range(self.n_elements):
                self.values_elements_cbu[i] = math.sqrt(math.log(self.total_count_cbu) / self.element_count_cbu[i])

    def cbv_model(self, success, repeat, chosen_combination, update):
        if update:
            if success == 1:
                if repeat == 1:
                    increment = 0
                else:
                    increment = 1
            else:
                if repeat == 1:
                    increment = 0
                else:
                    increment = -1

            self.values_elements_cbv[chosen_combination[0]] += increment
            self.values_elements_cbv[chosen_combination[1]] += increment

    def recency_model(self, chosen_combination, update):
        if update:
            self.total_count_rec += 1
            self.element_lastused_rec += 1
            self.element_lastused_rec[chosen_combination[0]] = 0
            self.element_lastused_rec[chosen_combination[1]] = 0
            for i in range(self.n_elements):
                self.values_elements_rec[i] = self.element_lastused_rec[i] / self.total_count_rec


class empowerment_value:
    def __init__(self):
        pass

    def standardize_prior(self, empowerment_values):
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(np.array(empowerment_values).reshape(-1, 1)).flatten()
        return normalized_values, scaler

    def de_normalize(self, value, scaler):
        return scaler.inverse_transform(np.array(value).reshape(-1, 1)).flatten()[0]

    def likelihood(self, success, repeat, empowerment, increase_factor=1.1, decrease_factor=0.9):
        if success == 1:
            if repeat == 0:
                return empowerment * increase_factor
            else:
                return empowerment
        else:
            if repeat == 0:
                return empowerment
            else:
                return empowerment * decrease_factor

    def dynamic_update(self, prior, success, repeat, first, second, increase_factor=1.1, decrease_factor=0.9):
        posterior = dict(prior)
        for el in [first, second]:
            if el in posterior:
                posterior[el] = self.likelihood(success, repeat, prior[el], increase_factor, decrease_factor)
            posterior[el] = max(0, posterior[el])
        return posterior


# Initialize classes
element_values = element_value()
empowerment_values = empowerment_value()

# Load prior empowerment and standardize
prior_empowerment = pd.read_csv('dataset/alchemy2_element_empowerment_dataframe.csv')
prior_empowerment_values = prior_empowerment['emp_value_predicted'].values
normalized_prior_empowerment, scaler = empowerment_values.standardize_prior(prior_empowerment_values)
prior_empowerment_dict = dict(zip(prior_empowerment['index'], normalized_prior_empowerment))

# Load data
data_codes = pd.read_csv('output/data/Llama3_8B_LLM_results(full_codes).csv')
# Filter by temperature = 1
#data_codes = data_codes[data_codes['temperature'] == 1]

# Add inventory_increase column
data_codes['inventory_increase'] = 0
data_codes.loc[data_codes['trial'] == 0, 'inventory_increase'] = data_codes.loc[data_codes['trial'] == 0, 'inventory'].apply(lambda x: 1 if x > 4 else 0)
data_codes.loc[data_codes['trial'] > 0, 'inventory_increase'] = data_codes['inventory'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)

# We'll prepare the repeat and inventory_elements columns before processing
final_rows = []

player_groups = data_codes.groupby('id')

for player_id, player_data in tqdm(player_groups, desc="Preprocessing Players"):
    player_data = player_data.sort_values('trial').copy()
    player_data['repeat'] = 0
    used_combinations = []

    # Reset the index so we can refer to previous rows easily
    player_data = player_data.reset_index(drop=True)

    for i, trial_row in player_data.iterrows():
        trial_idx = trial_row['trial']
        if trial_idx == 0:
            player_data.at[i, 'repeat'] = 0
        else:
            first = trial_row['first']
            second = trial_row['second']
            invalid_combination = pd.isna(first) or pd.isna(second)

            if not invalid_combination:
                combo = sorted([first, second])
                if combo not in used_combinations:
                    used_combinations.append(combo)
                    player_data.at[i, 'repeat'] = 0
                else:
                    player_data.at[i, 'repeat'] = 1
            else:
                # If invalid, set repeat to NaN (no updates anyway)
                player_data.at[i, 'repeat'] = np.nan

    # Now process each player's data for logging and updating
    element_values.reset_model('cbu')
    element_values.reset_model('cbv')
    element_values.reset_model('rec')
    player_empowerment_dict = prior_empowerment_dict.copy()

    for _, row in player_data.iterrows():
        trial = int(row["trial"])
        first_el = row["first"]
        second_el = row["second"]
        success = row["success"]
        repeat = row["repeat"]
        inventory_increase = row["inventory_increase"]
        inventory_elements = literal_eval(row['inventory_elements'])

        # Check if first_el or second_el is NaN
        invalid_combination = pd.isna(first_el) or pd.isna(second_el)

        if not invalid_combination:
            chosen_combination = [int(first_el), int(second_el)]
        else:
            chosen_combination = None

        # Log current values (no updates yet)
        for elem in inventory_elements:
            elem = int(elem)
            if chosen_combination is not None:
                choice_flag = 1 if elem in chosen_combination else 0
            else:
                # If invalid combination, choice is 0 for all elements
                choice_flag = 0

            cbu_val = element_values.values_elements_cbu[elem]
            cbv_val = element_values.values_elements_cbv[elem]
            rec_val = element_values.values_elements_rec[elem]
            emp_val = player_empowerment_dict.get(elem, 0.0)

            final_rows.append({
                "id": player_id,
                "trial": trial,
                "element": elem,
                "choice": choice_flag,
                "cbu_value": cbu_val,
                "cbv_value": cbv_val,
                "recency_value": rec_val,
                "empowerment_value": emp_val
            })

        # Only update if we have a valid combination
        if (chosen_combination is not None) and (not np.isnan(repeat)):
            element_values.cbu_model(chosen_combination, update=True)
            element_values.cbv_model(success, repeat, chosen_combination, update=True)
            element_values.recency_model(chosen_combination, update=True)
            updated_posterior = empowerment_values.bayesian_update(
                player_empowerment_dict,
                success,
                repeat,
                chosen_combination[0],
                chosen_combination[1]
            )
            player_empowerment_dict.update(updated_posterior)
        # If invalid, no updates made

# Convert to DataFrame and denormalize empowerment
final_df = pd.DataFrame(final_rows)
for index, row in final_df.iterrows():
    final_df.at[index, "empowerment_value"] = empowerment_values.de_normalize(row["empowerment_value"], scaler)

final_df.to_csv('output/data/Llama3_8B_element_value.csv', index=False)
print("Saved to output/data/Llama3_8B_element_value.csv")
