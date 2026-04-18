#!/usr/bin/env python3
"""Generate all figures for Progress 2 report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

output_dir = '/home/aqua/idsarcasm-reproduction/results/figures/'
os.makedirs(output_dir, exist_ok=True)

models = ['LR', 'NB', 'SVM']

# === Figure 1: Twitter F1 BoW vs TF-IDF ===
bow_t   = [0.7206, 0.6722, 0.6850]
tfidf_t = [0.7143, 0.5174, 0.6783]
x = np.arange(len(models)); w = 0.32

fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - w/2, bow_t, w, label='BoW', color='#60a5fa', edgecolor='#1e40af', lw=0.8)
b2 = ax.bar(x + w/2, tfidf_t, w, label='TF-IDF', color='#f97316', edgecolor='#c2410c', lw=0.8)
ax.set_ylabel('F1-Score'); ax.set_title('Perbandingan F1-Score pada Dataset Twitter')
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.4, 0.8)
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8)
plt.savefig(output_dir+'f1_twitter_bow_vs_tfidf.png'); plt.close()
print("✅ f1_twitter_bow_vs_tfidf.png")

# === Figure 2: Reddit F1 BoW vs TF-IDF ===
bow_r   = [0.4857, 0.4592, 0.4031]
tfidf_r = [0.4959, 0.3499, 0.4467]

fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - w/2, bow_r, w, label='BoW', color='#60a5fa', edgecolor='#1e40af', lw=0.8)
b2 = ax.bar(x + w/2, tfidf_r, w, label='TF-IDF', color='#f97316', edgecolor='#c2410c', lw=0.8)
ax.set_ylabel('F1-Score'); ax.set_title('Perbandingan F1-Score pada Dataset Reddit')
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.25, 0.55)
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8)
plt.savefig(output_dir+'f1_reddit_bow_vs_tfidf.png'); plt.close()
print("✅ f1_reddit_bow_vs_tfidf.png")

# === Figure 3: Reproduksi vs Paper (grouped) ===
paper_t = [0.7142, 0.6721, 0.6782]; repo_t = [0.7143, 0.5174, 0.6783]
paper_r = [0.4887, 0.4591, 0.4467]; repo_r = [0.4959, 0.3499, 0.4467]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ww = 0.3
for ax, pap, rep, title, ylim in [(axes[0], paper_t, repo_t, 'Dataset Twitter (TF-IDF)', (0.4,0.8)),
                                    (axes[1], paper_r, repo_r, 'Dataset Reddit (TF-IDF)', (0.25,0.55))]:
    ax.bar(x - ww/2, pap, ww, label='Paper', color='#a78bfa', edgecolor='#6d28d9', lw=0.8)
    ax.bar(x + ww/2, rep, ww, label='Reproduksi', color='#34d399', edgecolor='#059669', lw=0.8)
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('F1-Score'); ax.set_ylim(ylim); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i in range(len(models)):
        d = rep[i]-pap[i]; s = '+' if d>=0 else ''
        ax.text(x[i], max(pap[i],rep[i])+0.012, f'{s}{d:.4f}', ha='center', va='bottom', fontsize=8,
                color='#059669' if abs(d)<0.01 else '#dc2626')
fig.suptitle('Perbandingan F1-Score: Reproduksi vs Paper (TF-IDF)', fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(output_dir+'f1_reproduksi_vs_paper.png'); plt.close()
print("✅ f1_reproduksi_vs_paper.png")

# === Figure 4: Pipeline Architecture ===
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 3.5); ax.axis('off')

stages = [
    (0.5, 'Raw Text\n(CSV)', '#dbeafe', '#1e40af'),
    (2.5, 'Tokenisasi\n(nltk)', '#fef3c7', '#92400e'),
    (4.5, 'Vektorisasi\n(BoW/TF-IDF)', '#d1fae5', '#065f46'),
    (6.5, 'Model\n(LR/NB/SVM)', '#ede9fe', '#5b21b6'),
    (8.5, 'GridSearchCV\n+ Evaluasi', '#fce7f3', '#9d174d'),
]
subs = [
    'w11wo/twitter_indonesia_sarcastic', 'word_tokenize',
    'CountVectorizer /\nTfidfVectorizer', 'sklearn.LogisticRegression\n/ MultinomialNB / SVC',
    'Accuracy, Precision\nRecall, F1-Score'
]
arr = dict(arrowstyle='->', color='#6b7280', lw=1.5)
for i, (xp, lab, fc, ec) in enumerate(stages):
    r = mpatches.FancyBboxPatch((xp, 0.8), 1.6, 1.6, boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec, lw=1.5)
    ax.add_patch(r)
    ax.text(xp+0.8, 1.6, lab, ha='center', va='center', fontsize=9, fontweight='bold', color=ec)
    ax.text(xp+0.8, 0.25, subs[i], ha='center', va='center', fontsize=7, color='#6b7280', style='italic')
    if i < len(stages)-1: ax.annotate('', xy=(stages[i+1][0], 1.6), xytext=(xp+1.65, 1.6), arrowprops=arr)
ax.set_title('Arsitektur Pipeline Eksperimen Classical ML', fontsize=13, fontweight='bold', pad=10)
plt.savefig(output_dir+'pipeline_architecture.png'); plt.close()
print("✅ pipeline_architecture.png")

print(f"\n📁 All figures saved to: {output_dir}")
