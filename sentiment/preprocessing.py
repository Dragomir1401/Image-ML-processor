import re
import unicodedata
from collections import Counter
import numpy as np
import torch

class TextPreprocessor:
    def __init__(self, vocab_size=20000, max_len=200):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_built = False
    
    def clean_text(self, text):
        text = str(text).lower()
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def tokenize(self, text):
        text = self.clean_text(text)
        return text.split()
    
    def build_vocab(self, texts):
        print("Building vocabulary...")
        all_words = []
        for text in texts:
            tokens = self.tokenize(text)
            all_words.extend(tokens)
        
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word2idx)} words")
    
    def text_to_sequence(self, text):
        tokens = self.tokenize(text)
        sequence = [self.word2idx.get(word, 1) for word in tokens]
        return sequence
    
    def pad_sequence(self, sequence):
        if len(sequence) >= self.max_len:
            return sequence[:self.max_len]
        else:
            return sequence + [0] * (self.max_len - len(sequence))
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = self.text_to_sequence(text)
            seq = self.pad_sequence(seq)
            sequences.append(seq)
        return np.array(sequences)
    
    def get_vocab_size(self):
        return len(self.word2idx)

def load_fasttext_embeddings(embedding_file, word2idx, embedding_dim=300):
    print(f"Loading FastText embeddings from {embedding_file}...")
    embeddings = {}
    
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < embedding_dim + 1:
                    continue
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    embeddings[word] = vector
    except FileNotFoundError:
        print(f"Warning: {embedding_file} not found. Using random embeddings.")
        return None
    
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    found = 0
    for word, idx in word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
            found += 1
    
    print(f"Found {found}/{vocab_size} words in FastText embeddings")
    return embedding_matrix

class SimpleEmbedding:
    @staticmethod
    def create_embedding_layer(vocab_size, embedding_dim=300, pretrained_embeddings=None):
        if pretrained_embeddings is not None:
            embedding = torch.nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False
            )
        else:
            embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        return embedding
