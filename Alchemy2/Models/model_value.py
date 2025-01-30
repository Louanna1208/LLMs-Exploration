import pandas as pd
import os
import numpy as np
import math
import json
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
os.chdir('/Github/LLMs_game/Alchemy2')
from data_analysis.data_process import element_name_to_code, element_code_to_name, LLM_data_preprocessing
from Models.inventory import Inventory
import random
from tqdm import tqdm

class Alchemy2Model:
    def __init__(self):
        self.n_elements = 720
        self.data = []
        #load data
        self.probability_table = pd.read_hdf('dataset/alchemy2GametreeTable-data-crawl300.h5')
        self.empowerment_info = pd.read_csv('dataset/alchemy2EmpowermentTable-data-crawl300.csv')
        self.element_vectors = np.loadtxt('dataset/alchemy2ElementVectors-crawl300.txt')
        self.trueemp_table = pd.read_csv('dataset/alchemy2CombinationTrueemp.csv')
        self.combination_table_json = json.load(open('dataset/alchemy2CombinationTable.json'))
    
    def reset_model(self, model_type):
        """Resets model-specific parameters based on the model type."""
        if model_type == 'cbu':
            self.values_elements_cbu = np.zeros(self.n_elements)
            self.total_count_cbu = 1
            self.element_count_cbu = np.ones(self.n_elements)
        elif model_type == 'cbv':
            self.values_elements_cbv = np.zeros(self.n_elements)
            self.used_combinations = []
        elif model_type == 'rec':
            self.values_elements_rec = np.zeros(self.n_elements)
            self.total_count_rec = 1
            self.element_lastused_rec = np.zeros(self.n_elements)

    # Binary Model
    # the values are the probabilities of success
    def binary_model(self, combination):
        """
        Binary model calculates the predicted success probability for a combination.

        Args:
            combination (list): Combination of two element indices.

        Returns:
            float: Predicted success probability for the combination.
        """
        bin_value = self.probability_table.loc[tuple(combination), 'predSuccess']
        return bin_value

    # Count-Based Uncertainty (CBU) Model
    def cbu_model(self, combination, chosen_combination, success, update):
        """
        Count-Based Uncertainty (CBU) Model.

        Args:
            combination (list): Combination of two element indices to compute value.
            chosen_combination (list): Combination chosen by the player.
            reset (bool): If True, resets the model.

        Returns:
            float: Computed uncertainty value for the given combination.
        """
        
        cbu_value = self.values_elements_cbu[combination[0]] + self.values_elements_cbu[combination[1]]
        # Update model specifics
        if update:
            self.total_count_cbu += 1
            self.element_count_cbu[chosen_combination[0]] += 1
            self.element_count_cbu[chosen_combination[1]] += 1

            for i in range(self.n_elements):
                self.values_elements_cbu[i] = math.sqrt( 
                    math.log(self.total_count_cbu) / self.element_count_cbu[i]
                )
        return cbu_value

    # Count-Based Value (CBV) Model
    def cbv_model(self, combination, chosen_combination, success, update):
        """
        Count-Based Value (CBV) Model.

        Args:
            combination (list): Combination of two element indices to compute value.
            chosen_combination (list): Combination chosen by the player.
            results (int or list): Results of the combination (single int or list).
            reset (bool): If True, resets the model.

        Returns:
            float: Computed value for the given combination.
        """
        cbv_value = self.values_elements_cbv[combination[0]] + self.values_elements_cbv[combination[1]]
        comb = sorted(combination)
        # Update model specifics
        if update:
            # Ensure results is a list
            if success == 1:
                if comb not in self.used_combinations:# New results, increase values based on result length
                    self.used_combinations.append(comb)
                    self.values_elements_cbv[chosen_combination[0]] += 1
                    self.values_elements_cbv[chosen_combination[1]] += 1
                else:
                    self.values_elements_cbv[chosen_combination[0]] += 0
                    self.values_elements_cbv[chosen_combination[1]] += 0
            elif success == 0:# No new results, reduce values
                if comb not in self.used_combinations:
                    self.used_combinations.append(comb)
                    self.values_elements_cbv[chosen_combination[0]] -= 1
                    self.values_elements_cbv[chosen_combination[1]] -= 1
                else:
                    self.values_elements_cbv[chosen_combination[0]] -= 0
                    self.values_elements_cbv[chosen_combination[1]] -= 0

        return cbv_value

    
    #empowerment model
    def emp_model(self, combination):
        # Ensure the combination is sorted (to make order irrelevant)
        combination = sorted(combination)
        
        # Filter rows where the unordered pair matches
        emp_value = self.empowerment_info[
            ((self.empowerment_info['first'] == combination[0]) & (self.empowerment_info['second'] == combination[1])) |
            ((self.empowerment_info['first'] == combination[1]) & (self.empowerment_info['second'] == combination[0]))
        ]['empChild'].values
        return emp_value
    #empowerment direct model
    #Binary model based on self-constructed game tree.
    def empdirect_model(self, combination):
        empdirect_value = self.probability_table.loc[tuple(combination), 'predEmp']
        return empdirect_value
    
    #recency model
    #Values based on number of trials since the element was last used.
    def recency_model(self, combination, chosen_combination, success, update):
        """
        Recency Model.

        Args:
            combination (list): Combination of two element indices to compute value.
            chosen_combination (list): Combination chosen by the player.
            reset (bool): If True, resets the model.

        Returns:
            float: Computed recency value for the given combination.
        """
        recency_value = self.values_elements_rec[combination[0]] + self.values_elements_rec[combination[1]]
        # Update model specifics
        if update:
            self.total_count_rec += 1
            self.element_lastused_rec += 1
            self.element_lastused_rec[chosen_combination[0]] = 0    
            self.element_lastused_rec[chosen_combination[1]] = 0

            for i in range(self.n_elements):
                self.values_elements_rec[i] = self.element_lastused_rec[i] / self.total_count_rec
        return recency_value
    
    #similarity model
    #Values based on cosine similarity between element vectors.
    def sim_model(self, combination):
        similarities = cosine_similarity(self.element_vectors, self.element_vectors)
        sim_value = similarities[combination[0]][combination[1]]
        return sim_value
    
    #truebin model
    #Values based on true binary values.
    def truebin_model(self, combination):
        utility = 0
        if str(combination[0]) in self.combination_table_json and str(combination[1]) in self.combination_table_json[str(combination[0])]:
            utility = 1
        return utility
    
    #trueemp model
    #Empowerment model based on true game tree.
    def trueemp_model(self, combination):
        # check results of combination
        combination = sorted(combination)
        trueemp_value = self.trueemp_table[
            ((self.trueemp_table['Element1'] == combination[0]) & (self.trueemp_table['Element2'] == combination[1])) |
            ((self.trueemp_table['Element1'] == combination[1]) & (self.trueemp_table['Element2'] == combination[0]))
        ]['Trueemp'].values
        return trueemp_value
    
    
    def get_model_value(self, data, z_score=True, matched=True, model_type='gpt-4o'):
        """
        Compute values for all models, reset models per player, and store them.

        Args:
            data (pd.DataFrame): Input dataset.
            z_score (bool): Whether to compute z-scores for model deltas.

        Returns:
            pd.DataFrame: DataFrame containing computed model deltas.
        """
        inventory = Inventory()
        player_id = data.iloc[0]['id']  # Initialize player ID
        inventory.reset()

        models = [
            ('cbv', self.cbv_model),
            ('cbu', self.cbu_model),
            ('rec', self.recency_model),
            ('sim', self.sim_model),
            ('binary', self.binary_model),
            ('emp', self.emp_model),
            ('empdirect', self.empdirect_model),
            ('truebin', self.truebin_model),
            ('trueemp', self.trueemp_model),
        ]
        self.reset_model('cbv')
        self.reset_model('cbu')
        self.reset_model('rec')
        for idx, trial in tqdm(data.iterrows(), desc="Processing Trials", total=len(data)):
        #for idx, trial in data.iterrows():
            if trial['id'] != player_id:
                player_id = trial['id']
                inventory.reset()
                self.reset_model('cbv') #reset cbv model
                self.reset_model('cbu') #reset cbu model
                self.reset_model('rec') #reset rec model    
                print(f"Reset models for player {player_id}")
            
            # Parse results
            raw_results = trial.get('results', '[]')
            if isinstance(raw_results, int):
                results = [raw_results]
            elif isinstance(raw_results, list):
                results = raw_results
            else:
                results = []
            success = trial['success']
            # initialize dictionary to store info on combination trial
            if model_type in ['8B_intervention_empowerment', '8B_intervention_uncertainty', '70B_intervention_empowerment']:
                trial_data = {'model':model_type+'_'+str(trial['model']), 'temperature': trial['temperature'], 'id': player_id, 'trial': trial['trial'], 'inventory': trial['inventory']}
            else:
                trial_data = {'model': model_type, 'temperature': trial['temperature'], 'id': player_id, 'trial': trial['trial'], 'inventory': trial['inventory']}
            # Extract and sort chosen combination
            chosen_combination = sorted([int(trial['first']), int(trial['second'])])
            trial_data.update({'first': chosen_combination[0], 'second': chosen_combination[1]})
            # Determine random combination (for comparison)
            #New version with matching of the combination success:
            if matched == True:
                if str(chosen_combination[0]) in self.combination_table_json and \
                   str(chosen_combination[1]) in self.combination_table_json[str(chosen_combination[0])]:
                    random_combination = sorted([random.choice(inventory.inventory_successfull)])
                else:
                    random_combination = sorted([random.choice(inventory.inventory_not_successfull)])
                random_combination = sorted(random_combination[0])
            else:
                random_combination = sorted([random.choice(list(inventory.inventory_used)), random.choice(list(inventory.inventory_used))])

            # Simulate decision
            decision = random.randint(0, 1)
            trial_data.update({'decision': decision})

            # Compute values and deltas for all models
            for model_name, model_func in models:
                try:
                    if model_name in ['sim', 'binary', 'emp', 'empdirect', 'truebin', 'trueemp']:
                        # Models requiring only 'combination'
                        value_chosen = model_func(chosen_combination)
                        value_random = model_func(random_combination)
                        # if values are in the list, take the first element
                        if isinstance(value_chosen, np.ndarray):
                            value_chosen = value_chosen[0]
                        if isinstance(value_random, np.ndarray):
                            value_random = value_random[0]
                    elif model_name in ['cbv', 'cbu', 'rec']:
                        # Models requiring 'combination', 'chosen_combination', and 'results'
                        value_random = model_func(random_combination, random_combination, success, update=False)
                        value_chosen = model_func(chosen_combination, chosen_combination, success, update=True)
                        # if values are in the list, take the first element
                        if isinstance(value_chosen, np.ndarray):
                            value_chosen = value_chosen[0]
                        if isinstance(value_random, np.ndarray):
                            value_random = value_random[0]
                    else:
                        raise ValueError(f"Unhandled model: {model_name}")

                    # Compute delta
                    delta = value_chosen - value_random if decision == 1 else value_random - value_chosen
                    trial_data[f'value_{model_name}_chosen'] = value_chosen
                    trial_data[f'value_{model_name}_random'] = value_random
                    trial_data[f'delta_{model_name}'] = delta

                except Exception as e:
                    trial_data[f'delta_{model_name}'] = np.nan
                    tqdm.write(f"Error in {model_name} for trial {idx}: {e}")
            # for debugging
            #print(f"Trial {idx}, Chosen Combination: {chosen_combination}, Random Combination: {random_combination}") 
            #print("Inventory successfull:", inventory.inventory_successfull)
            #print("Inventory not successfull:", inventory.inventory_not_successfull)
            #print("Uncertainty:", trial_data[f'value_cbu_chosen'])
            #print("CBV:", trial_data[f'value_cbv_chosen'])
            #print("Recency:", trial_data[f'value_rec_chosen'])
            #print(f"Emp Value: {trial_data[f'value_emp_chosen']}, Random Emp Value: {trial_data[f'value_emp_random']}")
            #print(f"Type of Emp Value: {type(trial_data[f'value_emp_chosen'])}, Type of Random Emp Value: {type(trial_data[f'value_emp_random'])}")   
            #print(f"Delta for model {model_name}: {trial_data[f'delta_{model_name}']}")
            self.data.append(trial_data)

            # Update inventory and cbv, cbu, rec models
            # Update inventory based on the current trial
            results = inventory.update(results)
            inventory.update_success_list()  # Refresh success and failure lists
            #self.cbv_model(chosen_combination, chosen_combination, results, update=True)
            #self.cbu_model(chosen_combination, chosen_combination, results, update=True)
            #self.recency_model(chosen_combination, chosen_combination, results, update=True)


        # Convert to DataFrame
        results_df = pd.DataFrame(self.data)

        # Normalize deltas
        if z_score is True:
            for model_name, _ in models:
                results_df['delta_{}'.format(model_name)] = zscore(results_df['delta_{}'.format(model_name)], ddof=1, nan_policy="omit")

        return results_df


def main(dataset, model_type):
    model = Alchemy2Model()
    results = model.get_model_value(dataset, z_score=False, matched=False, model_type=model_type)
    # Save results if the file does not exist
    output_file_path = f'output/alchemy2_{model_type}_values(not_matched).csv'
    if not os.path.exists(output_file_path):
        results.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}.")

if __name__ == "__main__":
    #Base data
    Base_LLM_data_path = 'output/data/base_gpt-4o-2024-08-06_500_results.json'
    Base_LLM_with_names, Base_LLM_with_codes = LLM_data_preprocessing(Base_LLM_data_path,'gpt-4o')
    
    #Llama3 data
    Llama3_70B_LLM_data_path = 'output/data/llama3_70B_500_results.json'
    Llama3_70B_LLM_with_names, Llama3_70B_LLM_with_codes = LLM_data_preprocessing(Llama3_70B_LLM_data_path,'Llama3')

    #Llama3 data
    Llama3_8B_LLM_data_path = 'output/data/llama3_8B_500_results.json'
    Llama3_8B_LLM_with_names, Llama3_8B_LLM_with_codes = LLM_data_preprocessing(Llama3_8B_LLM_data_path,'Llama3')


    # o1 data
    o1_LLM_data_path = 'output/data/base_o1-2024-12-17_500_results.json'
    o1_LLM_with_names, o1_LLM_with_codes = LLM_data_preprocessing(o1_LLM_data_path,'gpt-4o')

    # deepseek-reasoner data
    deepseek_reasoner_data_path = 'output/data/reasoning_deepseek-reasoner_500_results.json'
    deepseek_reasoner_data_with_names, deepseek_reasoner_data_with_codes = LLM_data_preprocessing(deepseek_reasoner_data_path,'gpt-4o')

    # 8B intervention data
    intervention_8B_uncertainty_data_path = 'output/data/8B_intervention_uncertainty_500_results.json'
    intervention_8B_uncertainty_data_with_names, intervention_8B_uncertainty_data_with_codes = LLM_data_preprocessing(intervention_8B_uncertainty_data_path,'LLaMA3.1_intervention')
    intervention_8B_empowerment_data_path = 'output/data/8B_intervention_empowerment_500_results.json'
    intervention_8B_empowerment_data_with_names, intervention_8B_empowerment_data_with_codes = LLM_data_preprocessing(intervention_8B_empowerment_data_path,'LLaMA3.1_intervention')

    # 70B intervention data
    intervention_70B_empowerment_data_path = 'output/data/70B_intervention_empowerment_500_results.json'
    intervention_70B_empowerment_data_with_names, intervention_70B_empowerment_data_with_codes = LLM_data_preprocessing(intervention_70B_empowerment_data_path,'LLaMA3.1_intervention')


    #main(Base_LLM_with_codes, 'Base_gpt-4o')
    #main(Llama3_70B_LLM_with_codes, 'Llama3_70B')
    #main(Llama3_8B_LLM_with_codes, 'Llama3_8B')
    #main(o1_LLM_with_codes, 'o1')
    #main(deepseek_reasoner_data_with_codes, 'deepseek-reasoner')
    #main(intervention_8B_uncertainty_data_with_codes, '8B_intervention_uncertainty')
    #main(intervention_8B_empowerment_data_with_codes, '8B_intervention_empowerment')
    main(intervention_70B_empowerment_data_with_codes, '70B_intervention_empowerment')