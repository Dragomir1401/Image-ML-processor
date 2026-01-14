import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import copy
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                                       weight_decay=config.get('weight_decay', 0))
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                                      momentum=0.9, weight_decay=config.get('weight_decay', 0))
        
        if config.get('scheduler') == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
                                              patience=5)
        elif config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        else:
            self.scheduler = None
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        self.best_model_wts = None
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.config["epochs"]} - '
                      f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% - '
                      f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% - '
                      f'LR: {current_lr:.6f}')
        
        elapsed_time = time.time() - start_time
        print(f'\nTraining complete in {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s')
        print(f'Best Val Acc: {self.best_val_acc:.2f}%')
        
        self.model.load_state_dict(self.best_model_wts)
        
        return self.history
    
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
