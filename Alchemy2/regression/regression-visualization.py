import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
# Set the working directory to the folder containing the CSV file
os.chdir(r'\Github\LLMs_game\Alchemy2\regression')
#------------------------------------------------------------------------------------------------
#plot the regression results by temperature and model
#------------------------------------------------------------------------------------------------
# Load the regression results data
file_path = 'results/regression_results_summary(not_matched).csv'
# which including two sub-datasets: whole and split
regression_data = pd.read_csv(file_path)
# Get the two sub-datasets
whole_data = regression_data[regression_data['temperature'].isna()]
split_data = regression_data[regression_data['temperature'].notna()]

datasets = ['LLaMA3.1-8B', 'LLaMA3.1-70B', 'gpt-4o','Human','deepseek-reasoner','o1']
colors = {
    'LLaMA3.1-8B': '#84BA42',
    'LLaMA3.1-70B': '#DBB428',
    'gpt-4o': '#7ABBDB',
    'Human': '#A51C36',
    'deepseek-reasoner': '#bcfce7',
    'o1': '#682478'
}

# Filter LLM data for temperature-specific results
llm_temp_data = split_data[split_data['temperature'].notna()]

# Extract human data for a horizontal line
human_data = whole_data[whole_data['dataset'] == 'Human']
# o1 data
o1_data = whole_data[whole_data['dataset'] == 'o1']
# deepseek-reasoner data
deepseek_reasoner_data = whole_data[whole_data['dataset'] == 'deepseek-reasoner']

# Initialize the figure with 5 subplots empowerment, uncertainty, trial, empowerment:trial, uncertainty:trial
#subplot_terms = ['emp', 'cbu', 'trial', 'emp:trial', 'cbu:trial']
subplot_terms = ['emp', 'cbu']

# change the title of the subplots
#subplot_titles = ['Empowerment', 'Uncertainty', 'Trial', 'Empowerment:Trial', 'Uncertainty:Trial']
subplot_titles = ['Empowerment', 'Uncertainty']
#fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
axes = axes.flatten()

# Remove extra subplot (6th)
#fig.delaxes(axes[-1])

# Define LLM models
llm_models = ['LLaMA3.1-8B', 'LLaMA3.1-70B','gpt-4o']

# Create subplots for each term
for i, term in enumerate(subplot_terms):
    ax = axes[i]
    # Plot each LLM model's temperature data with error bars
    for j, model in enumerate(llm_models):
        if model == 'gpt-4o':
            label = "GPT-4o"
        elif model == 'LLaMA3.1-8B':
            label = "LLaMA3.1-8B"
        elif model == 'LLaMA3.1-70B':
            label = "LLaMA3.1-70B"
        else:
            label = model
        model_data = llm_temp_data[
            (llm_temp_data['term'] == term) & 
            (llm_temp_data['dataset'].str.contains(model))
        ]
        sns.lineplot(
            data=model_data,
            x='temperature', y='estimate', marker='o', ax=ax, label=label, color=colors[model], linewidth=2
        )
        # Add error bars
        #ax.errorbar(
        #       model_data['temperature'], model_data['estimate'], yerr=1.96*model_data['std_error'],
        #       fmt='none', ecolor='black', elinewidth=1.5, capsize=5
        #)


    # Add horizontal line for human baseline
    human_estimate = human_data[human_data['term'] == term]['estimate'].values[0]
    ax.axhline(y=human_estimate, color='#A51C36', linestyle='--', label='Human')
    # deepseek-reasoner data
    deepseek_reasoner_estimate = deepseek_reasoner_data[deepseek_reasoner_data['term'] == term]['estimate'].values[0]
    ax.axhline(y=deepseek_reasoner_estimate, color='#bcfce7', linestyle='-.', label='DeepSeek-R1')
    # o1 data
    o1_estimate = o1_data[o1_data['term'] == term]['estimate'].values[0]
    ax.axhline(y=o1_estimate, color='#682478', linestyle='-.', label='o1')


    # Set subplot titles and labels
    ax.set_title(subplot_titles[i], fontsize=12)
    #set x-tick labels 0, 0.3, 0.7, 1.0
    ax.set_xticks([0, 0.3, 0.7, 1.0])
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylim(-6, 6)
    ax.set_ylabel("Average Estimate",fontsize=12)
    ax.legend(title="Model",loc='lower right',fontsize=8)
    # Explicitly set x-tick and y-tick label font sizes for each subplot
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
# add title to the figure
plt.suptitle("Regression estimates by temperature and model", y=0, fontsize=12)
#plt.savefig('picture/regression_estimates_by_temperature_and_model(not_matched).png', dpi=300)
#plt.savefig('picture/regression_estimates_by_temperature_and_model(not_matched).pdf', dpi=300)
plt.savefig('picture/regression_estimates_by_temperature_and_model(not_matched)_add.png', dpi=300)
#plt.savefig('picture/regression_estimates_by_temperature_and_model(not_matched)_add.pdf', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------
#plot the regression results after intervention
#------------------------------------------------------------------------------------------------

def plot_intervention_estimates(model_name, data):
    # Extract categories and data
    categories = ['empowerment', 'uncertainty']
    # set color for original(LlaMA 8B #84BA42, LlaMA 70B #DBB428), empowerment_intervention(#C7C1DE), uncertainty_intervention(#BD7795)
    if model_name == 'LLaMA3.1-8B':
        colors = ['#84BA42', '#C7C1DE', '#BD7795']
    elif model_name == 'LLaMA3.1-70B':
        colors = ['#DBB428', '#C7C1DE', '#BD7795']
    else:
        colors = ['#84BA42', '#DBB428', '#FFEE6F', '#BD7795']
    conditions = list(data.keys())
    values = np.array([[data[cond][cat] for cat in categories] for cond in conditions])

    # Plotting the bar chart
    x = np.arange(len(categories))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, condition in enumerate(conditions):
        ax.bar(x + i * width, values[i], width, label=condition, color=colors[i])

    # Adding labels and title
    ax.set_xlabel('Exploration',fontsize=14)
    ax.set_ylabel('Estimate',fontsize=14)
    ax.set_title(f'{model_name} Intervention Estimates',fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories,fontsize=14)
    ax.legend(fontsize=14)

    # Display the chart
    plt.tight_layout()
    plt.savefig(f'picture/{model_name}_intervention_regression_results.png', dpi=300)
    plt.savefig(f'picture/{model_name}_intervention_regression_results.pdf', dpi=300)
    plt.show()

LLaMA8B_data = {
    'original': {'empowerment': -0.074, 'uncertainty': 0.241},
    'empowerment_intervention': {'empowerment': -0.238, 'uncertainty': 0.559},
    'uncertainty_intervention': {'empowerment': -0.140, 'uncertainty': -1.082},
}
plot_intervention_estimates('LLaMA3.1-8B', LLaMA8B_data)

LLaMA70B_data = {
    'original': {'empowerment': 0.150, 'uncertainty': 1.814},
    'empowerment_intervention': {'empowerment': 0.039, 'uncertainty': 1.547}
}
plot_intervention_estimates('LLaMA3.1-70B', LLaMA70B_data)