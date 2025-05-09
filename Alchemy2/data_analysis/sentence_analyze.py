import os
import json
import pandas as pd # Used for the user's original data loading structure
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize # Keep word_tokenize import from user's code
import openai
import jsonlines
os.chdir(r'\Github\LLMs_game_copy\Alchemy2')

# --- Your Defined Tags ---
REASONING_TAGS = [
    "state_goal",
    "check_current_inventory",
    "past_trial_analysis",
    "element_property_reasoning",
    "combination_analysis",
    "outcome_prediction",
    "final_choice"
]

# --- LLM Prompt Template (for single sentence labeling) ---
System_Prompt = """
You are an expert AI assistant for analyzing problem-solving thought processes. Your task is to carefully read a reasoning passage from a Little Alchemy 2 player, identify a specific sentence within it by its index, and assign a single label to *that specific sentence* based on its primary role in the overall thought process context.

Use the following 7 labels and their precise definitions:

1.  **state_goal**: The sentence explicitly mentions the overall objective of the game (creating new elements) or the high-level strategy being followed (e.g., "try basics first", "systematic approach").
2.  **check_current_inventory**: The sentence lists or directly refers to the elements the player currently possesses.
3.  **past_trial_analysis**: The sentence refers to a *specific past attempt* by trial number or explicitly states the *known outcome* (success or failure) of a combination that was tried previously. This is about retrieving and stating known historical results.
4.  **element_property_reasoning**: The sentence discusses the inherent characteristics, qualities, or real-world behavior of elements to explain why a combination might work or what its nature is.
5.  **combination_analysis**: The sentence proposes a specific pair of elements for combination *as an idea* to consider, OR discusses the combination in a general sense *before* or *without* specifically checking its history or predicting its outcome. This is about the initial idea generation or general consideration of a pair. *Prioritize assigning 'past_trial_analysis', 'outcome_prediction', or 'element_property_reasoning' if the sentence's main focus fits one of those specific categories better.*
6.  **outcome_prediction**: The AI speculates or predicts what a combination (especially an untried one) might create as a result.
7.  **final_choice**: The AI clearly and definitively states the specific combination that the AI has decided to attempt in the current game step.

**Instructions:**

* Read the entire "Reasoning Passage" provided below to understand the full context.
* Identify the sentence corresponding to the specified "Sentence Index".
* Assign **exactly one** label from the list (1-7) that *most accurately* reflects the primary contribution of *that specific sentence* to the overall reasoning flow.
* If the identified sentence seems relevant to multiple labels, choose the single label representing its main purpose in that specific context. Refer back to the precise definitions.
* Output **only** the assigned label string. Do not include the sentence index, the sentence text, any JSON formatting, or any other words. Just the label string.
"""

# ——————— API Wrapper ————————
openai.api_key = "Your API Key"

def call_API(prompt, config):
    resp = openai.chat.completions.create(
        model=config['model'],
        messages=[
            {"role": "system", "content": System_Prompt},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=50,
        temperature=config['temperature'],
        seed=config['seed'],
    )
    return resp.choices[0].message.content.strip(), resp.usage

def prompt_single_sentence(reasoning_text, sentence):
    return f"""
**Reasoning Passage:**
{reasoning_text}

**Sentence to Label:** {sentence}
"""

# ——————— Data Processing ————————
def process_reasoning(model_data, model_name):
    """
    model_data: raw[model_name], which is { player_id: { trial_id: {…,'reason':str,…} } }
    """
    rows = []
    for player_id, trials in model_data.items():
        for trial_id, trial_data in trials.items():
            reason = trial_data.get('reason')
            if not reason: 
                continue
            for i, sent in enumerate(sent_tokenize(reason)):
                rows.append({
                    'model': model_name,
                    'player_id': str(player_id),
                    'trial_id': str(trial_id),
                    'sentence_index': i,
                    'sentence': sent
                })
    df = pd.DataFrame(rows)
    print(f"Processed {model_name}: {df.shape[0]} sentences")
    return df

def label_sentences(raw_model_data, df_sentences, config, save_path):
    already_labeled = set()
    if os.path.exists(save_path):
        with jsonlines.open(save_path, mode='r') as reader:
            for obj in reader:
                key = (obj['player_id'], obj['trial_id'], obj['sentence_index'])
                already_labeled.add(key)

    with jsonlines.open(save_path, mode='a') as writer:
        for _, row in df_sentences.iterrows():
            key = (row['player_id'], row['trial_id'], row['sentence_index'])
            if key in already_labeled:
                continue

            reason = raw_model_data[row['player_id']][row['trial_id']]['reason']
            prompt = prompt_single_sentence(reason, row['sentence'])
            try:
                label, usage = call_API(prompt, config)
            except Exception as e:
                print(f"API Error on {key}: {e}")
                continue

            result = {
                **row.to_dict(),
                'label': label
            }

            writer.write(result)
            print(f"{key} → {label}")
            print(f"Token usage: {usage}")

# ---------------------
# Main Execution
# ---------------------
def main():
    #model_name = 'deepseek-reasoner'
    #json_path = 'output/data/reasoning_deepseek-reasoner_500_results.json'
    model_name = 'gpt-4o-2024-08-06'
    json_path = 'output/data/reasoning_gpt-4o-2024-08-06_500_results.json'
    #save_path = 'output/data/sentence_labels_deepseek-reasoner.jsonl'
    save_path = 'output/data/sentence_labels_gpt-4o-2024-08-06.jsonl'
    config = {
        'model': 'gpt-4.1-2025-04-14',
        'temperature': 0.0,
        'seed': 42
    }

    os.makedirs('output/data', exist_ok=True)

    with open(json_path, 'r') as f:
        raw = json.load(f)

    if model_name not in raw:
        raise ValueError(f"Top-level key '{model_name}' not found in the file.")
    # only test in trial_id 0-149
    model_data = {k: v for k, v in raw[model_name].items() if int(k) <= 149}
    #model_data = raw[model_name]
    sentence_df = process_reasoning(model_data, model_name)
    label_sentences(model_data, sentence_df, config, save_path)

if __name__ == "__main__":
    main()
