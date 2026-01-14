import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path

def load_data(data_dir='sentiment_data'):
    train_df = pd.read_csv(Path(data_dir) / 'train.csv')
    test_df = pd.read_csv(Path(data_dir) / 'test.csv')
    return train_df, test_df

def analyze_class_distribution(train_df, test_df, output_dir='results/sentiment_analysis'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    train_counts = train_df['label'].value_counts().sort_index()
    test_counts = test_df['label'].value_counts().sort_index()
    
    labels = ['Negative', 'Positive']
    
    axes[0].bar(labels, train_counts.values, color=['coral', 'lightgreen'], alpha=0.8)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Training Set - Class Distribution', fontsize=13, fontweight='bold')
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontsize=11)
    
    axes[1].bar(labels, test_counts.values, color=['coral', 'lightgreen'], alpha=0.8)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Test Set - Class Distribution', fontsize=13, fontweight='bold')
    for i, v in enumerate(test_counts.values):
        axes[1].text(i, v, str(v), ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Class Distribution:")
    print(f"\nTraining Set:")
    print(f"  Negative: {train_counts[0]} ({train_counts[0]/len(train_df)*100:.2f}%)")
    print(f"  Positive: {train_counts[1]} ({train_counts[1]/len(train_df)*100:.2f}%)")
    print(f"\nTest Set:")
    print(f"  Negative: {test_counts[0]} ({test_counts[0]/len(test_df)*100:.2f}%)")
    print(f"  Positive: {test_counts[1]} ({test_counts[1]/len(test_df)*100:.2f}%)")

def compute_text_lengths(df):
    df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
    df['num_chars'] = df['text'].apply(lambda x: len(str(x)))
    return df

def plot_text_length_distribution(train_df, test_df, output_dir='results/sentiment_analysis'):
    train_df = compute_text_lengths(train_df)
    test_df = compute_text_lengths(test_df)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for label in [0, 1]:
        label_name = 'Negative' if label == 0 else 'Positive'
        color = 'coral' if label == 0 else 'lightgreen'
        
        train_subset = train_df[train_df['label'] == label]
        test_subset = test_df[test_df['label'] == label]
        
        axes[0, label].hist(train_subset['num_words'], bins=50, color=color, alpha=0.7, edgecolor='black')
        axes[0, label].set_xlabel('Number of Words', fontsize=11)
        axes[0, label].set_ylabel('Frequency', fontsize=11)
        axes[0, label].set_title(f'Train - {label_name} - Word Count Distribution', fontsize=12, fontweight='bold')
        axes[0, label].axvline(train_subset['num_words'].mean(), color='red', linestyle='--', 
                               label=f'Mean: {train_subset["num_words"].mean():.1f}')
        axes[0, label].legend()
        
        axes[1, label].hist(train_subset['num_chars'], bins=50, color=color, alpha=0.7, edgecolor='black')
        axes[1, label].set_xlabel('Number of Characters', fontsize=11)
        axes[1, label].set_ylabel('Frequency', fontsize=11)
        axes[1, label].set_title(f'Train - {label_name} - Character Count Distribution', fontsize=12, fontweight='bold')
        axes[1, label].axvline(train_subset['num_chars'].mean(), color='red', linestyle='--',
                               label=f'Mean: {train_subset["num_chars"].mean():.1f}')
        axes[1, label].legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("\nText Length Statistics:")
    for label in [0, 1]:
        label_name = 'Negative' if label == 0 else 'Positive'
        train_subset = train_df[train_df['label'] == label]
        print(f"\n{label_name} Reviews:")
        print(f"  Words - Mean: {train_subset['num_words'].mean():.2f}, "
              f"Median: {train_subset['num_words'].median():.2f}, "
              f"Max: {train_subset['num_words'].max()}")
        print(f"  Chars - Mean: {train_subset['num_chars'].mean():.2f}, "
              f"Median: {train_subset['num_chars'].median():.2f}, "
              f"Max: {train_subset['num_chars'].max()}")

def get_most_common_words(texts, n=20):
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', str(text).lower())
        all_words.extend(words)
    
    return Counter(all_words).most_common(n)

def plot_most_common_words(train_df, output_dir='results/sentiment_analysis'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, label in enumerate([0, 1]):
        label_name = 'Negative' if label == 0 else 'Positive'
        color = 'coral' if label == 0 else 'lightgreen'
        
        subset = train_df[train_df['label'] == label]
        common_words = get_most_common_words(subset['text'], n=15)
        
        words, counts = zip(*common_words)
        
        axes[idx].barh(range(len(words)), counts, color=color, alpha=0.8)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words)
        axes[idx].set_xlabel('Frequency', fontsize=11)
        axes[idx].set_title(f'Top 15 Words - {label_name} Reviews', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        
        for i, v in enumerate(counts):
            axes[idx].text(v, i, f' {v}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'most_common_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_sentiment_data(data_dir='sentiment_data', output_dir='results/sentiment_analysis'):
    print("=" * 60)
    print("RO_SENT DATASET ANALYSIS")
    print("=" * 60)
    
    train_df, test_df = load_data(data_dir)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    print("\n--- Class Distribution ---")
    analyze_class_distribution(train_df, test_df, output_dir)
    
    print("\n--- Text Length Analysis ---")
    plot_text_length_distribution(train_df, test_df, output_dir)
    
    print("\n--- Most Common Words ---")
    plot_most_common_words(train_df, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete. Results saved in: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    analyze_sentiment_data()
