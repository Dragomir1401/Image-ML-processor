import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class Evaluator:
    def __init__(self, model, test_loader, device, classes):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.classes = classes
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        metrics = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc * 100
        }
        
        return metrics, all_preds, all_labels
    
    def print_metrics(self, metrics):
        print('\n' + '='*60)
        print('TEST SET EVALUATION')
        print('='*60)
        print(f'Accuracy:  {metrics["accuracy"]:.2f}%')
        print(f'Precision: {metrics["precision"]:.2f}%')
        print(f'Recall:    {metrics["recall"]:.2f}%')
        print(f'F1-Score:  {metrics["f1"]:.2f}%')
        print('\nPer-class Accuracy:')
        for i, cls in enumerate(self.classes):
            print(f'  {cls}: {metrics["per_class_accuracy"][i]:.2f}%')
        print('='*60)
