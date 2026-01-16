import torch
import torch.nn as nn
from pathlib import Path
from sentiment import get_sentiment_loaders, SimpleRNN, SentimentLSTM
from training import Trainer, Evaluator
from utils import plot_training_history, plot_confusion_matrix, save_results_table

def train_sentiment_model(model_type, config, output_dir='results/sentiment_experiments'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'\nLoading sentiment dataset...')
    train_loader, test_loader, preprocessor = get_sentiment_loaders(
        data_dir='sentiment_data',
        batch_size=config['batch_size'],
        vocab_size=config['vocab_size'],
        max_len=config['max_len'],
        augment=config.get('augment', False)
    )
    
    vocab_size = preprocessor.get_vocab_size()
    print(f'Vocabulary size: {vocab_size}')
    
    pretrained_embeddings = None
    
    if model_type == 'rnn':
        model = SimpleRNN(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            pretrained_embeddings=pretrained_embeddings
        )
    else:
        model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config.get('bidirectional', False),
            pretrained_embeddings=pretrained_embeddings
        )
    
    print(f'\nModel architecture ({model_type.upper()}):')
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    print('\n' + '='*60)
    print('TRAINING')
    print('='*60)
    
    trainer = Trainer(model, train_loader, test_loader, device, config)
    history = trainer.train()
    
    model_dir = Path(output_dir) / 'models'
    model_path = model_dir / f'{config["name"]}_{model_type}.pth'
    trainer.save_model(model_path)
    print(f'\nModel saved to {model_path}')
    
    plot_training_history(
        history,
        save_path=Path(output_dir) / 'plots' / f'{config["name"]}_{model_type}_training.png'
    )
    
    print('\n' + '='*60)
    print('EVALUATION ON TEST SET')
    print('='*60)
    
    evaluator = Evaluator(model, test_loader, device, ['Negative', 'Positive'])
    metrics, preds, labels = evaluator.evaluate()
    evaluator.print_metrics(metrics)
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        ['Negative', 'Positive'],
        save_path=Path(output_dir) / 'plots' / f'{config["name"]}_{model_type}_confusion_matrix.png'
    )
    
    return {
        'model': model_type.upper(),
        'config': config['name'],
        'accuracy': f"{metrics['accuracy']:.2f}",
        'precision': f"{metrics['precision']:.2f}",
        'recall': f"{metrics['recall']:.2f}",
        'f1': f"{metrics['f1']:.2f}",
        'best_val_acc': f"{trainer.best_val_acc:.2f}"
    }

def main():
    results = []
    
    rnn_configs = [
        {
            'name': 'RNN_baseline',
            'vocab_size': 20000,
            'max_len': 200,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'optimizer': 'adam',
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'epochs': 20,
            'scheduler': 'plateau',
            'augment': False
        },
        {
            'name': 'RNN_deep',
            'vocab_size': 20000,
            'max_len': 200,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout': 0.3,
            'optimizer': 'adam',
            'lr': 0.0003,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'epochs': 20,
            'scheduler': 'plateau',
            'augment': False
        }
    ]
    
    lstm_configs = [
        {
            'name': 'LSTM_baseline',
            'vocab_size': 20000,
            'max_len': 200,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': False,
            'optimizer': 'adam',
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'epochs': 20,
            'scheduler': 'plateau',
            'augment': False
        },
        {
            'name': 'LSTM_bidirectional',
            'vocab_size': 20000,
            'max_len': 200,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'bidirectional': True,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'epochs': 20,
            'scheduler': 'plateau',
            'augment': False
        },
        {
            'name': 'LSTM_with_augmentation',
            'vocab_size': 20000,
            'max_len': 200,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'bidirectional': True,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'epochs': 20,
            'scheduler': 'plateau',
            'augment': True
        }
    ]
    
    print('\n' + '#'*60)
    print('TRAINING RNN MODELS')
    print('#'*60)
    
    for config in rnn_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        result = train_sentiment_model('rnn', config)
        results.append(result)
    
    print('\n\n' + '#'*60)
    print('TRAINING LSTM MODELS')
    print('#'*60)
    
    for config in lstm_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        result = train_sentiment_model('lstm', config)
        results.append(result)
    
    save_results_table(results, 'results/sentiment_experiments/sentiment_results_summary.csv')

if __name__ == '__main__':
    main()
