import torch
import argparse
from pathlib import Path
from data import get_data_loaders
from models import SimpleCNN, CNN
from training import Trainer, Evaluator
from utils import plot_training_history, plot_confusion_matrix, save_results_table

def train_cnn(dataset_path, dataset_name, config, output_dir, model_type='simple'):
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
    
    num_classes = len(classes)
    
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {classes}')
    
    if model_type == 'simple':
        model = SimpleCNN(
            num_classes=num_classes,
            dropout=config['dropout'],
            use_batchnorm=config['use_batchnorm']
        )
    else:
        model = CNN(
            num_classes=num_classes,
            dropout=config['dropout'],
            use_batchnorm=config['use_batchnorm']
        )
    
    print(f'\nModel architecture ({model_type} CNN):')
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
    model_path = model_dir / f'{dataset_name}_cnn.pth'
    trainer.save_model(model_path)
    print(f'\nModel saved to {model_path}')
    
    plot_training_history(
        history,
        save_path=Path(output_dir) / 'plots' / f'{dataset_name}_cnn_training.png'
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
        save_path=Path(output_dir) / 'plots' / f'{dataset_name}_cnn_confusion_matrix.png'
    )
    
    return {
        'dataset': dataset_name,
        'model': f'CNN_{model_type}',
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
            'name': 'SimpleCNN_baseline',
            'model_type': 'simple',
            'dropout': 0.3,
            'use_batchnorm': False,
            'optimizer': 'adam',
            'lr': 0.003,
            'weight_decay': 0.0,
            'batch_size': 128,
            'epochs': 15,
            'scheduler': None,
            'augment': False
        },
        {
            'name': 'SimpleCNN_with_batchnorm',
            'model_type': 'simple',
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'adam',
            'lr': 0.003,
            'weight_decay': 1e-4,
            'batch_size': 128,
            'epochs': 25,
            'scheduler': 'plateau',
            'augment': False
        },
        {
            'name': 'SimpleCNN_optimized',
            'model_type': 'simple',
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'adam',
            'lr': 0.003,
            'weight_decay': 1e-4,
            'batch_size': 128,
            'epochs': 30,
            'scheduler': 'plateau',
            'augment': True
        }
    ]
    
    print('\n' + '#'*60)
    print('TRAINING IMAGEBITS WITH CNN')
    print('#'*60)
    
    for config in imagebits_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        model_type = config.pop('model_type')
        
        result = train_cnn(
            'imagebits/imagebits',
            f'imagebits_{config["name"]}',
            config,
            'results/cnn_experiments',
            model_type=model_type
        )
        results.append(result)
    
    land_patches_configs = [
        {
            'name': 'SimpleCNN_baseline',
            'model_type': 'simple',
            'dropout': 0.3,
            'use_batchnorm': False,
            'optimizer': 'adam',
            'lr': 0.003,
            'weight_decay': 0.0,
            'batch_size': 128,
            'epochs': 15,
            'scheduler': None,
            'augment': False,
            'target_size': (96, 96)
        },
        {
            'name': 'SimpleCNN_optimized',
            'model_type': 'simple',
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'adam',
            'lr': 0.003,
            'weight_decay': 1e-4,
            'batch_size': 128,
            'epochs': 20,
            'scheduler': 'plateau',
            'augment': True,
            'target_size': (96, 96)
        }
    ]
    
    print('\n\n' + '#'*60)
    print('TRAINING LAND PATCHES WITH CNN')
    print('#'*60)
    
    for config in land_patches_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        model_type = config.pop('model_type')
        
        result = train_cnn(
            'land_patches/land_patches',
            f'land_patches_{config["name"]}',
            config,
            'results/cnn_experiments',
            model_type=model_type
        )
        results.append(result)
    
    save_results_table(results, 'results/cnn_experiments/cnn_results_summary.csv')

if __name__ == '__main__':
    main()
