import random
import re

class TextAugmentation:
    def __init__(self, p=0.1):
        self.p = p
    
    def random_swap(self, words, n=1):
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words
    
    def random_delete(self, words):
        if len(words) == 1:
            return words
        
        new_words = []
        for word in words:
            if random.random() > self.p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return [random.choice(words)]
        
        return new_words
    
    def synonym_replacement(self, words, n=1):
        return words
    
    def augment_text(self, text, num_aug=1):
        text = str(text) if text is not None else ""
        words = text.split()
        
        if len(words) == 0:
            return [text]
        
        augmented_texts = []
        for _ in range(num_aug):
            aug_words = words.copy()
            
            if random.random() < 0.5:
                aug_words = self.random_swap(aug_words, n=max(1, len(words) // 10))
            
            if random.random() < 0.3:
                aug_words = self.random_delete(aug_words)
            
            augmented_texts.append(' '.join(aug_words))
        
        return augmented_texts
