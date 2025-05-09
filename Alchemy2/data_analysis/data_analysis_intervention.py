# Step 1: Import necessary libraries and data processing module
import json
import numpy as np
import os
os.chdir('D:/Github/LLMs_game/Alchemy2')
import pandas as pd
from data_analysis.data_process import LLM_data_preprocessing, Human_data_preprocessing
from data_analysis.data_process import element_name_to_code, element_code_to_name
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind


# Step 2: Call data_processing function to load and process data
#Llama3 data
Llama3_70B_LLM_data_path = 'output/data/Llama3_70B_500_results.json'
Llama3_70B_LLM_with_names, Llama3_70B_LLM_with_codes = LLM_data_preprocessing(Llama3_70B_LLM_data_path,'Llama3')

# Llama3 8B
Llama3_8B_LLM_data_path = 'output/data/Llama3_8B_500_results.json'
Llama3_8B_LLM_with_names, Llama3_8B_LLM_with_codes = LLM_data_preprocessing(Llama3_8B_LLM_data_path,'Llama3')

# Llama3.1 70B intervention data（uncertainty）
intervention_70B_uncertainty_data_path = 'output/data/70B_intervention_uncertainty_500_results.json'
intervention_70B_uncertainty_names, intervention_70B_uncertainty_codes = LLM_data_preprocessing(intervention_70B_uncertainty_data_path,'LLaMA3.1_intervention')
#intervention_70B_uncertainty_names.to_csv('output/data/70B_intervention_uncertainty_500_results.csv', index=False)

# Llama3.1 70B intervention data（empowerment）
intervention_70B_empowerment_data_path = 'output/data/70B_intervention_empowerment_500_results.json'
intervention_70B_empowerment_names, intervention_70B_empowerment_codes = LLM_data_preprocessing(intervention_70B_empowerment_data_path,'LLaMA3.1_intervention')

#Llama3.1 8B intervention data（uncertainty）
intervention_8B_uncertainty_data_path = 'output/data/8B_intervention_uncertainty_500_results.json'
intervention_8B_uncertainty_names, intervention_8B_uncertainty_codes = LLM_data_preprocessing(intervention_8B_uncertainty_data_path,'LLaMA3.1_intervention')

#Llama3.1 8B intervention data（empowerment）
intervention_8B_empowerment_data_path = 'output/data/8B_intervention_empowerment_500_results.json'
intervention_8B_empowerment_names, intervention_8B_empowerment_codes = LLM_data_preprocessing(intervention_8B_empowerment_data_path,'LLaMA3.1_intervention')


def fill_missing_inventory(df, trial_num):
    filled_dfs = []
    for uid, user_df in df.groupby('id'):
        user_df = user_df.set_index('trial').sort_index()
        # create a full index
        full_index = pd.Index(range(trial_num), name='trial')
        # rebuild the DataFrame, fill the missing
        user_df = user_df.reindex(full_index)
        user_df['id'] = uid
        # fill the missing inventory with the previous value, if the first item is still missing, fill it with 4
        user_df['inventory'] = user_df['inventory'].ffill().fillna(4)
        filled_dfs.append(user_df.reset_index())
    return pd.concat(filled_dfs, ignore_index=True)

#--------------------------------------------------------------------------------------------------
# figure 1: plot the average inventory of each intervention
#--------------------------------------------------------------------------------------------------
def plot_average_inventory_of_each_intervention(model_name, intervention_name, original_data, intervention_data):
    # set the trial number
    trial_num = 500
    plt.figure(figsize=(8, 6))
    Llama3_original = original_data[original_data['temperature'] == 1.0]
    Llama3_original = fill_missing_inventory(Llama3_original, trial_num)

    # get the data for each intervention
    intervention_group = intervention_data.groupby('model')
    for intervention in intervention_group.groups:
        intervention_data = intervention_group.get_group(intervention) # get the data for each intervention
        # get the each intervention's final inventory max, min, mean, and confidence intervals
        final_inventory = intervention_data.groupby('id')['inventory'].max()
        print(f'{intervention_name} = {intervention} Maximum inventory size: {final_inventory.max()}')   
        print(f'{intervention_name} = {intervention} Minimum inventory size: {final_inventory.min()}')
        print(f'{intervention_name} = {intervention} Mean inventory size: {final_inventory.mean()}')
        print(f'{intervention_name} = {intervention} Confidence Intervals:{sm.stats.DescrStatsW(final_inventory).tconfint_mean(alpha = 0.05)}')
        intervention_data = fill_missing_inventory(intervention_data, trial_num)
        intervention_data = intervention_data.groupby('trial')['inventory'].mean()[:trial_num] # get the mean inventory for each trial
        # print each trial's mean inventory
        print(f'{intervention_name} = {intervention} Mean inventory for each trial: {intervention_data}')
        plt.plot(intervention_data, label=f'{intervention_name}(intervention = {intervention})')
    # plot the original data
    plt.plot(Llama3_original.groupby('trial')['inventory'].mean()[:trial_num], label=f'{model_name}(original)', color='#DBB428', linewidth=2)
    if intervention_name == 'empowerment':
        plt.title(f'{model_name} Average Inventory of Empowerment Intervention', fontsize=16)
    else:
        plt.title(f'{model_name} Average Inventory of Uncertainty Intervention', fontsize=16)
    plt.xlabel('Trial', fontsize=16)
    plt.ylabel('Average Inventory', fontsize=16)
    # set x-tick labels
    plt.xticks(fontsize=14)
    # set y-tick labels
    plt.yticks(fontsize=14)
    plt.xlim(0, trial_num)
    if model_name == 'LLaMA3.1-70B':
        plt.ylim(0, 50)
    else:
        plt.ylim(0, 30)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig(f'output/picture/{model_name}_average_inventory_of_{intervention_name}_intervention_full.png', dpi=300)
    # save figure as pdf for latex
    #plt.savefig(f'output/picture/{model_name}_average_inventory_of_each_intervention.pdf', format='pdf',dpi=300)
    plt.show()

plot_average_inventory_of_each_intervention('LLaMA3.1-70B', 'uncertainty', Llama3_70B_LLM_with_codes, intervention_70B_uncertainty_codes)
plot_average_inventory_of_each_intervention('LLaMA3.1-70B', 'empowerment', Llama3_70B_LLM_with_codes, intervention_70B_empowerment_codes)
plot_average_inventory_of_each_intervention('LLaMA3.1-8B', 'uncertainty', Llama3_8B_LLM_with_codes, intervention_8B_uncertainty_codes)
plot_average_inventory_of_each_intervention('LLaMA3.1-8B', 'empowerment', Llama3_8B_LLM_with_codes, intervention_8B_empowerment_codes)

#--------------------------------------------------------------------------------------------------
# figure 2: plot the average inventory of each intervention
#--------------------------------------------------------------------------------------------------
def inventory_of_interventions(model_name, original_data, empowerment_data, uncertainty_data):
    # set the trial number
    trial_num = 500
    plt.figure(figsize=(8, 6))
    Llama3_original = original_data[original_data['temperature'] == 1.0]
    Llama3_original = fill_missing_inventory(Llama3_original, trial_num)
    empowerment_data = empowerment_data[empowerment_data['model'] == '0.0']
    empowerment_data = fill_missing_inventory(empowerment_data, trial_num)
    uncertainty_data = uncertainty_data[uncertainty_data['model'] == '0.0']
    uncertainty_data = fill_missing_inventory(uncertainty_data, trial_num)

    # set color for original(LlaMA 8B #84BA42, LlaMA 70B #DBB428), empowerment_intervention(#C7C1DE), uncertainty_intervention(#BD7795)
    if model_name == 'LLaMA3.1-8B':
        plt.plot(Llama3_original.groupby('trial')['inventory'].mean()[:trial_num], label=f'{model_name} Original', color='#84BA42', linewidth=2)
    else:
        plt.plot(Llama3_original.groupby('trial')['inventory'].mean()[:trial_num], label=f'{model_name} Original', color='#DBB428', linewidth=2)
    plt.plot(empowerment_data.groupby('trial')['inventory'].mean()[:trial_num], label=f'{model_name} Empowerment', color='#C7C1DE', linewidth=2)
    plt.plot(uncertainty_data.groupby('trial')['inventory'].mean()[:trial_num], label=f'{model_name} Uncertainty', color='#BD7795', linewidth=2)
    plt.title(f'{model_name} Average Inventory of Interventions', fontsize=16)
    plt.xlabel('Trial', fontsize=16)
    plt.ylabel('Average Inventory', fontsize=16)
    # set x-tick labels
    plt.xticks(fontsize=14)
    # set y-tick labels
    plt.yticks(fontsize=14)
    plt.xlim(0, trial_num)
    if model_name == 'LLaMA3.1-70B':
        plt.ylim(0, 50)
    else:
        plt.ylim(0, 30)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig(f'output/picture/{model_name}_average_inventory_of_interventions_full.png', dpi=300)
    # save figure as pdf for latex
    #plt.savefig(f'output/picture/{model_name}_average_inventory_of_each_intervention.pdf', format='pdf',dpi=300)
    plt.show()

inventory_of_interventions('LLaMA3.1-8B', Llama3_8B_LLM_with_codes, intervention_8B_empowerment_codes, intervention_8B_uncertainty_codes)
inventory_of_interventions('LLaMA3.1-70B', Llama3_70B_LLM_with_codes, intervention_70B_empowerment_codes, intervention_70B_uncertainty_codes)

#--------------------------------------------------------------------------------------------------
# figure 3: plot the condition percentages of each model
#--------------------------------------------------------------------------------------------------
class ModelConditionTracker:
    def __init__(self, data, model_name, trial_num, condition):
        self.model_name = model_name
        self.data = data
        self.trial_num = trial_num
        self.condition = condition

        self.conditions = {
            "Failure with Existing Combination",
            "Failure with New Combination",
            "Success with Existing Combination",
            "Success with New Combination",
            "Invalid Trial"
        }

    def _categorize_trials_for_id(self, data):
        """Categorize trials for a single ID."""
        all_trial_numbers = set(range(self.trial_num))
        categorized_trials = []
    
        for id, player_data in data.groupby('id'):
            unique_pairs = set()
            successful_pairs = set()
            exist_trial_numbers = set(player_data['trial'])
            missing_trial_numbers = all_trial_numbers - exist_trial_numbers

            # Record failed trials
            if missing_trial_numbers:
                for missing_trial in missing_trial_numbers:
                    categorized_trials.append({'trial': missing_trial, 'condition': "Invalid Trial"})

            for _, row in player_data.iterrows():
                trial_number = row['trial']
                pair = tuple(sorted([row['first'], row['second']]))

                if row['success'] == 1:
                    if pair in successful_pairs:
                        condition = "Success with Existing Combination"
                    else:
                        successful_pairs.add(pair)
                        condition = "Success with New Combination"
                else:
                    if pair in unique_pairs:
                        condition = "Failure with Existing Combination"
                    else:
                        unique_pairs.add(pair)
                        condition = "Failure with New Combination"
                
                categorized_trials.append({'trial': trial_number, 'condition': condition})

        return categorized_trials

    def calculate_model_percentages(self, data):
        """Calculate percentages for Human or Large LLM."""
        if data is None:
            return None
        id_percentages = []
        
        for id_val in data['id'].unique():
            id_data = data[data['id'] == id_val]
            conditions = self._categorize_trials_for_id(id_data)
            total_trials = len(conditions)
            id_dict = {}
            
            for condition in self.conditions:
                count = sum(1 for trial in conditions if trial['condition'] == condition)
                percentage = count / total_trials * 100
                id_dict[condition] = percentage

            id_percentages.append(id_dict)
        
        percentages_df = pd.DataFrame(id_percentages)
        return {
            'mean': percentages_df.mean(),
            'std': percentages_df.std(),
            'n_ids': len(data['id'].unique())
        }

    def count_trials_per_interval(self, interval_size=30):
        """Count trials by conditions in intervals of specified size."""
        results = []

        for dataset in self.data['dataset'].unique():
            dataset_data = self.data[self.data['dataset'] == dataset]

            for intervention in dataset_data['model'].unique():
                intervention_data = dataset_data[dataset_data['model'] == intervention]

                for id_val in intervention_data['id'].unique():
                    id_data = intervention_data[intervention_data['id'] == id_val]
                    categorized_trials = self._categorize_trials_for_id(id_data)

                    max_trial = self.trial_num
                    intervals = range(0, max_trial, interval_size)

                    for start in intervals:
                        end = start + interval_size

                        # Filter trials within the current interval
                        interval_trials = [
                            trial for trial in categorized_trials
                            if start <= trial['trial'] < end
                        ]

                        # Initialize counts for all conditions in this interval
                        condition_counts = {condition: 0 for condition in self.conditions}

                        # Count the trials per condition
                        for trial in interval_trials:
                            condition = trial['condition']
                            condition_counts[condition] += 1

                        # Add results to the DataFrame
                        for condition, count in condition_counts.items():
                            results.append({
                                "dataset": dataset,
                                "intervention": intervention,
                                "interval": f"{start}-{end-1}",
                                "condition": condition,
                                "count": count
                            })

        return pd.DataFrame(results)


    def plot_trials(self, interval_size=30):
        """Plot the average trial counts by conditions for each dataset and intervention."""
        trials_df = self.count_trials_per_interval(interval_size=interval_size)

        # Filter the data for the specified condition
        condition_data = trials_df[trials_df['condition'] == self.condition]

        # Group by dataset, intervention, and interval, then average the counts
        averaged_data = (
            condition_data
            .groupby(['dataset', 'intervention', 'interval'], as_index=False)['count']
            .mean()  # Take the average across IDs
        )

        # Extract numerical start value from interval and sort
        averaged_data['interval_start'] = averaged_data['interval'].str.split('-').str[0].astype(int)
        averaged_data = averaged_data.sort_values(by='interval_start')

        # Plot each dataset separately
        plt.figure(figsize=(12, 8))
        for dataset in averaged_data['dataset'].unique():
            dataset_data = averaged_data[averaged_data['dataset'] == dataset]
            for intervention in dataset_data['intervention'].unique():
                intervention_data = dataset_data[dataset_data['intervention'] == intervention]

                # Dynamically set the label
                if dataset == 'original':
                    label = f'{self.model_name} (original)'
                else:
                    label = f'{self.model_name} ({dataset} = {intervention})'

                # Plot the data
                plt.plot(
                    intervention_data['interval_start'],
                    intervention_data['count'],
                    marker='o',
                    label=label
                )

        # Add plot details
        plt.title(f'{self.model_name} Trials Per Interval for {self.condition}', fontsize=16)
        plt.xlabel('Trial Interval Start', fontsize=14)
        plt.ylabel('Average Trial Count', fontsize=14)
        plt.xticks(rotation=45, fontsize=10)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.savefig(f'output/picture/{self.model_name}_trials_per_interval_{self.condition.replace(" ", "_")}.pdf', format='pdf', dpi=300)
        plt.show()


# Add labels to distinguish datasets
Llama3_70B_original = Llama3_70B_LLM_with_codes[Llama3_70B_LLM_with_codes['temperature'] == 1.0]
Llama3_8B_original = Llama3_8B_LLM_with_codes[Llama3_8B_LLM_with_codes['temperature'] == 1.0]
Llama3_70B_original['dataset'] = 'original'
Llama3_8B_original['dataset'] = 'original'
intervention_8B_uncertainty_codes['dataset'] = 'uncertainty_intervention'
intervention_8B_empowerment_codes['dataset'] = 'empowerment_intervention'
intervention_70B_uncertainty_codes['dataset'] = 'uncertainty_intervention'
intervention_70B_empowerment_codes['dataset'] = 'empowerment_intervention'


# Combine all datasets
all_data_8B = pd.concat([Llama3_8B_original, intervention_8B_uncertainty_codes, intervention_8B_empowerment_codes])
all_data_70B = pd.concat([Llama3_70B_original, intervention_70B_uncertainty_codes, intervention_70B_empowerment_codes])

tracker = ModelConditionTracker(
    data=all_data_8B, 
    model_name='LLaMA3.1-8B', 
    trial_num=500, 
    condition="Invalid Trial"
)
tracker.plot_trials(interval_size=30)

tracker = ModelConditionTracker(
    data=all_data_70B, 
    model_name='LLaMA3.1-70B', 
    trial_num=500, 
    condition="Invalid Trial"
)
tracker.plot_trials(interval_size=30)
