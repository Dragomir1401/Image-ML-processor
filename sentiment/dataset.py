import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sentiment.preprocessing import TextPreprocessor

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, augment=False, augmenter=None):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.augment = augment
        self.augmenter = augmenter
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        label = self.labels[idx]
        
        if self.augment and self.augmenter and random.random() < 0.3:
            aug_texts = self.augmenter.augment_text(text, num_aug=1)
            text = aug_texts[0]
        
        sequence = self.preprocessor.text_to_sequence(text)
        sequence = self.preprocessor.pad_sequence(sequence)
        
        return torch.LongTensor(sequence), torch.tensor(label, dtype=torch.long)

def load_sentiment_data(data_dir='sentiment_data'):
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    train_df = train_df.dropna(subset=['text'])
    test_df = test_df.dropna(subset=['text'])
    
    return train_df, test_df

def get_sentiment_loaders(data_dir='sentiment_data', batch_size=64, vocab_size=20000, 
                          max_len=200, augment=False):
    train_df, test_df = load_sentiment_data(data_dir)
    
    preprocessor = TextPreprocessor(vocab_size=vocab_size, max_len=max_len)
    preprocessor.build_vocab(train_df['text'].values)
    
    augmenter = None
    if augment:
        from sentiment.augmentation import TextAugmentation
        augmenter = TextAugmentation(p=0.1)
    
    train_dataset = SentimentDataset(
        train_df['text'].values,
        train_df['label'].values,
        preprocessor,
        augment=augment,
        augmenter=augmenter
    )
    
    test_dataset = SentimentDataset(
        test_df['text'].values,
        test_df['label'].values,
        preprocessor,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, preprocessor
