
import os, json, jsonlines, itertools, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from transformers import AutoTokenizer
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir(r'Github\LLMs_game_copy\Alchemy2')
plt.rcParams["font.size"] = 12

LABEL_FILE_deepseek = 'output/data/sentence_labels_deepseek-reasoner.jsonl'
LABEL_FILE_gpt = 'output/data/sentence_labels_gpt-4o-2024-08-06.jsonl'
NGRAM_K = [3, 4, 5]          # The length of n-gram
TOP_K   = 10                 # The top k templates for each n

# ---------- Customized behavior order ----------
ordered_labels = [
    'state_goal',
    'check_current_inventory',
    'past_trial_analysis',
    'element_property_reasoning',
    'combination_analysis',
    'outcome_prediction',
    'final_choice',
]

ordered_labels_heatmap = [
    'begin',
    'state_goal',
    'check_current_inventory',
    'past_trial_analysis',
    'element_property_reasoning',
    'combination_analysis',
    'outcome_prediction',
    'final_choice',
    'end'
]

# ---------------- 1. Read + Compress ----------------
df_deepseek = pd.read_json(LABEL_FILE_deepseek, lines=True)
df_gpt = pd.read_json(LABEL_FILE_gpt, lines=True)

df_deepseek = df_deepseek[df_deepseek['label'].isin(ordered_labels)].copy()
df_deepseek.sort_values(['player_id','trial_id','sentence_index'], inplace=True)

df_gpt = df_gpt[df_gpt['label'].isin(ordered_labels)].copy()
df_gpt.sort_values(['player_id','trial_id','sentence_index'], inplace=True)

def compress(seq):
    """Remove consecutive duplicate labels"""
    out = []
    for lab in seq:
        if not out or lab != out[-1]:
            out.append(lab)
    return out

def add_begin_end(seq):
    return ['begin'] + seq + ['end']

seqs_deepseek = df_deepseek.groupby(['player_id', 'trial_id'])['label'].apply(list).map(compress).map(add_begin_end)
seqs_gpt = df_gpt.groupby(['player_id', 'trial_id'])['label'].apply(list).map(compress).map(add_begin_end)

seqs_deepseek_heatmap = df_deepseek.groupby(['player_id', 'trial_id'])['label'].apply(list).map(compress).map(add_begin_end)
seqs_gpt_heatmap = df_gpt.groupby(['player_id', 'trial_id'])['label'].apply(list).map(compress).map(add_begin_end)

# ---------- Heatmap ----------
idx_heatmap = {lab: i for i, lab in enumerate(ordered_labels_heatmap)}
count_mat_deepseek_heatmap = np.zeros((len(ordered_labels_heatmap), len(ordered_labels_heatmap)), dtype=int)
for seq in seqs_deepseek_heatmap:
    for a, b in zip(seq, seq[1:]):
        count_mat_deepseek_heatmap[idx_heatmap[a], idx_heatmap[b]] += 1

count_mat_gpt_heatmap = np.zeros((len(ordered_labels_heatmap), len(ordered_labels_heatmap)), dtype=int)
for seq in seqs_gpt_heatmap:
    for a, b in zip(seq, seq[1:]):
        count_mat_gpt_heatmap[idx_heatmap[a], idx_heatmap[b]] += 1

row_sums_deepseek_heatmap = count_mat_deepseek_heatmap.sum(axis=1, keepdims=True)
prob_mat_deepseek_heatmap = np.divide(count_mat_deepseek_heatmap, row_sums_deepseek_heatmap, where=row_sums_deepseek_heatmap!=0)

row_sums_gpt_heatmap = count_mat_gpt_heatmap.sum(axis=1, keepdims=True)
prob_mat_gpt_heatmap = np.divide(count_mat_gpt_heatmap, row_sums_gpt_heatmap, where=row_sums_gpt_heatmap!=0)
prob_mat_gpt_heatmap[row_sums_gpt_heatmap.flatten() == 0, :] = 0

# ---------- Draw probability Heatmap ----------
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'wspace': 0.05})
vmax = max(prob_mat_deepseek_heatmap.max(), prob_mat_gpt_heatmap.max())
cmap = 'YlGnBu'

for ax, prob_mat, title in zip(
    axs,
    [prob_mat_gpt_heatmap, prob_mat_deepseek_heatmap],
    ["GPT-4o", "DeepSeek-R1"]
):
    im = ax.imshow(prob_mat, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(ordered_labels_heatmap)))
    ax.set_xticklabels(ordered_labels_heatmap, rotation=40, ha='right', fontsize=13)
    ax.set_yticks(range(len(ordered_labels_heatmap)))
    ax.set_yticklabels(ordered_labels_heatmap, fontsize=13)
    ax.set_title(f"{title}", fontsize=16, pad=15)
    ax.set_xticks(np.arange(-.5, len(ordered_labels_heatmap), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ordered_labels_heatmap), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='both', which='both', length=0)
    for i in range(len(ordered_labels_heatmap)):
        for j in range(len(ordered_labels_heatmap)):
            # Only show text for nonzero probabilities
            if prob_mat[i, j] > 1e-8:
                ax.text(j, i, f"{prob_mat[i, j]:.2f}", ha='center', va='center', fontsize=11,
                        color='black' if prob_mat[i, j] < vmax*0.6 else 'white')

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.15)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.set_ylabel('Transition Probability', rotation=270, labelpad=18, fontsize=14)
cbar.ax.tick_params(labelsize=12)

fig.suptitle("Transition Probability Heatmap Comparison", fontsize=20, y=1.02)
plt.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.12, wspace=0.05)
plt.savefig('output/picture/transition_prob_heatmap_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✅ transition_prob_heatmap_comparison.png saved")

# ---------------- 3. High-frequency n‑gram templates ----------------
def ngram_iter(seq, n):
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i:i+n])

ngram_counts_deepseek = {n: collections.Counter() for n in [5, 10, 15, 20]}
for seq in seqs_deepseek:
    for n in [5, 10, 15, 20]:
        ngram_counts_deepseek[n].update(ngram_iter(seq, n))

ngram_counts_gpt = {n: collections.Counter() for n in [2, 3]}
for seq in seqs_gpt:
    for n in [2, 3]:
        ngram_counts_gpt[n].update(ngram_iter(seq, n))

print("\n=== DeepSeek n‑gram templates ===")
for n in [5, 10, 15, 20]:
    print(f"\nTop {TOP_K} for n={n}")
    for gram, cnt in ngram_counts_deepseek[n].most_common(TOP_K):
        print(f"{' → '.join(gram):50s}  {cnt}")

print("\n=== GPT-4o n-gram ===")
for n in [2, 3]:
    print(f"\nTop for n={n}")
    for gram, cnt in ngram_counts_gpt[n].most_common(TOP_K):
        print(f"{' → '.join(gram):50s}  {cnt}")

# ---------------- 5. Sequence length and unique label count comparison ----------------
def trial_stats(df, compressed=False):
    grouped = df.groupby(['player_id', 'trial_id'])
    if compressed:
        # For DeepSeek: use compressed label sequence length
        seq_lens = grouped['label'].apply(lambda x: len(compress(list(x)))).values
        unique_lens = grouped['label'].apply(lambda x: len(set(compress(list(x))))).values
    else:
        # For GPT-4o: use original sequence length
        seq_lens = grouped.size().values
        unique_lens = grouped['label'].nunique().values
    return seq_lens, unique_lens

seq_lens_gpt, unique_lens_gpt = trial_stats(df_gpt, compressed=False)
seq_lens_deepseek, unique_lens_deepseek = trial_stats(df_deepseek, compressed=True)

# Prepare means and SEMs for bar chart
N_gpt = len(seq_lens_gpt)
N_deepseek = len(seq_lens_deepseek)
seqlen_means = [np.mean(seq_lens_gpt), np.mean(seq_lens_deepseek)]
seqlen_sems = [np.std(seq_lens_gpt, ddof=1)/np.sqrt(N_gpt), np.std(seq_lens_deepseek, ddof=1)/np.sqrt(N_deepseek)]
uniq_means = [np.mean(unique_lens_gpt), np.mean(unique_lens_deepseek)]
uniq_sems = [np.std(unique_lens_gpt, ddof=1)/np.sqrt(N_gpt), np.std(unique_lens_deepseek, ddof=1)/np.sqrt(N_deepseek)]

labels_bar = ['GPT-4o', 'DeepSeek-R1']
x = np.arange(len(labels_bar))
width = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

# Subplot 1: Reasoning Length
bar1 = ax1.bar(x, seqlen_means, width, yerr=seqlen_sems, capsize=8, color='skyblue')
ax1.set_ylabel('Per-Trial Number of Sentences', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_bar)
ax1.set_ylim(0, 400)
#ax1.set_title('Reasoning Length')
ax1.tick_params(axis='y', labelsize=12)

# Subplot 2: Unique Labels
bar2 = ax2.bar(x, uniq_means, width, yerr=uniq_sems, capsize=8, color='coral')
ax2.set_ylabel('Per-Trial Number of Unique Labels', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_bar)
ax2.set_ylim(0, 8)
#ax2.set_title('Unique Label Count')
ax2.tick_params(axis='y', labelsize=12)

fig.suptitle('Per-Trial Reasoning Depth Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.savefig('output/picture/reasoning_depth_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------- 6. Token-level label length comparison ----------------
# Choose a large-model tokenizer (DeepSeek or Llama-2)
try:
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-base')
except Exception:
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

def get_token_count(text):
    return len(tokenizer.encode(str(text)))

# For GPT-4o: add token counts for all sentences using the new tokenizer
df_gpt['tokens'] = df_gpt['sentence'].apply(get_token_count)

# For DeepSeek: only count the first sentence of each compressed label block per trial using the new tokenizer
compressed_rows = []
for (player_id, trial_id), group in df_deepseek.groupby(['player_id', 'trial_id']):
    labels = group['label'].tolist()
    sentences = group['sentence'].tolist()
    # compress labels and keep the first sentence for each block
    compressed = []
    last_label = None
    for i, lab in enumerate(labels):
        if lab != last_label:
            compressed.append((lab, sentences[i]))
            last_label = lab
    for lab, sent in compressed:
        compressed_rows.append({
            'player_id': player_id,
            'trial_id': trial_id,
            'label': lab,
            'sentence': sent,
            'tokens': get_token_count(sent)
        })

df_deepseek_compressed = pd.DataFrame(compressed_rows)

# --- Update token-level label length comparison plot to use per-trial label total tokens ---
def per_trial_label_total_tokens_sem(df):
    # For each trial, sum token count for each label
    trial_label_sums = df.groupby(['player_id', 'trial_id', 'label'])['tokens'].sum().reset_index()
    # Now, for each label, aggregate across all trials
    stats = trial_label_sums.groupby('label')['tokens']
    means = stats.mean()
    stds = stats.std(ddof=1)
    ns = stats.count()
    sems = stds / np.sqrt(ns)
    return means, sems

means_gpt, sems_gpt = per_trial_label_total_tokens_sem(df_gpt)
means_deepseek, sems_deepseek = per_trial_label_total_tokens_sem(df_deepseek_compressed)

labels = ordered_labels
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, [means_gpt.get(l, 0) for l in labels], width, yerr=[sems_gpt.get(l, 0) for l in labels], label='GPT-4o', capsize=5, color='#4485C7')
ax.bar(x + width/2, [means_deepseek.get(l, 0) for l in labels], width, yerr=[sems_deepseek.get(l, 0) for l in labels], label='DeepSeek-R1', capsize=5, color='#bcfce7')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('Token Count')
ax.set_title('Per-Trial Total Token Count by Label')
ax.legend()
plt.tight_layout()
#plt.savefig('output/picture/token_count_by_label.png', dpi=300, bbox_inches='tight')
plt.show()


