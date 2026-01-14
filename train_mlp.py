import torch
import argparse
from pathlib import Path
from data import get_data_loaders
from models import MLP
from training import Trainer, Evaluator
from utils import plot_training_history, plot_confusion_matrix, save_results_table

def train_mlp(dataset_path, dataset_name, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'\nLoading {dataset_name} dataset...')
    train_loader, test_loader, val_loader, classes = get_data_loaders(
        dataset_path,
        batch_size=config['batch_size'],
        num_workers=2,
        target_size=config.get('target_size'),
        augment=config.get('augment', False)
    )
    
    if val_loader is None:
        val_loader = test_loader
    
    sample_input = next(iter(train_loader))[0]
    input_size = sample_input.view(sample_input.size(0), -1).size(1)
    num_classes = len(classes)
    
    print(f'Input size: {input_size}')
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {classes}')
    
    model = MLP(
        input_size=input_size,
        hidden_sizes=config['hidden_sizes'],
        num_classes=num_classes,
        dropout=config['dropout'],
        use_batchnorm=config['use_batchnorm']
    )
    
    print(f'\nModel architecture:')
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    print('\n' + '='*60)
    print('TRAINING')
    print('='*60)
    
    trainer = Trainer(model, train_loader, val_loader, device, config)
    history = trainer.train()
    
    model_dir = Path(output_dir) / 'models'
    model_path = model_dir / f'{dataset_name}_mlp.pth'
    trainer.save_model(model_path)
    print(f'\nModel saved to {model_path}')
    
    plot_training_history(
        history,
        save_path=Path(output_dir) / 'plots' / f'{dataset_name}_mlp_training.png'
    )
    
    print('\n' + '='*60)
    print('EVALUATION ON TEST SET')
    print('='*60)
    
    evaluator = Evaluator(model, test_loader, device, classes)
    metrics, preds, labels = evaluator.evaluate()
    evaluator.print_metrics(metrics)
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        classes,
        save_path=Path(output_dir) / 'plots' / f'{dataset_name}_mlp_confusion_matrix.png'
    )
    
    return {
        'dataset': dataset_name,
        'model': 'MLP',
        'config': str(config),
        'accuracy': f"{metrics['accuracy']:.2f}",
        'precision': f"{metrics['precision']:.2f}",
        'recall': f"{metrics['recall']:.2f}",
        'f1': f"{metrics['f1']:.2f}",
        'best_val_acc': f"{trainer.best_val_acc:.2f}"
    }

def main():
    results = []
    
    imagebits_configs = [
        {
            'name': 'MLP_baseline',
            'hidden_sizes': [512, 256],
            'dropout': 0.3,
            'use_batchnorm': False,
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 0.0,
            'batch_size': 256,
            'epochs': 15,
            'scheduler': None,
            'augment': False
        },
        {
            'name': 'MLP_with_batchnorm',
            'hidden_sizes': [512, 256],
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'batch_size': 256,
            'epochs': 15,
            'scheduler': 'plateau',
            'augment': False
        }
    ]
    
    print('\n' + '#'*60)
    print('TRAINING IMAGEBITS WITH MLP')
    print('#'*60)
    
    for config in imagebits_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        result = train_mlp(
            'imagebits/imagebits',
            f'imagebits_{config["name"]}',
            config,
            f'results/mlp_experiments'
        )
        results.append(result)
    
    land_patches_configs = [
        {
            'name': 'MLP_baseline',
            'hidden_sizes': [512, 256],
            'dropout': 0.3,
            'use_batchnorm': False,
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 0.0,
            'batch_size': 256,
            'epochs': 15,
            'scheduler': None,
            'augment': False,
            'target_size': (96, 96)
        },
        {
            'name': 'MLP_with_batchnorm',
            'hidden_sizes': [512, 256],
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'batch_size': 256,
            'epochs': 15,
            'scheduler': 'plateau',
            'augment': False,
            'target_size': (96, 96)
        }
    ]
    
    print('\n\n' + '#'*60)
    print('TRAINING LAND PATCHES WITH MLP')
    print('#'*60)
    
    for config in land_patches_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        result = train_mlp(
            'land_patches/land_patches',
            f'land_patches_{config["name"]}',
            config,
            f'results/mlp_experiments'
        )
        results.append(result)
    
    save_results_table(results, 'results/mlp_experiments/mlp_results_summary.csv')

if __name__ == '__main__':
    main()
