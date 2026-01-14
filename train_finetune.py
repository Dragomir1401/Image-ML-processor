import torch
from pathlib import Path
from data import get_data_loaders
from models import SimpleCNN
from training import Trainer, Evaluator
from utils import plot_training_history, plot_confusion_matrix, save_results_table

def fine_tune_model(pretrained_model_path, target_dataset_path, dataset_name, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'\nLoading pretrained model from {pretrained_model_path}...')
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    
    print(f'\nLoading {dataset_name} dataset...')
    train_loader, test_loader, val_loader, classes = get_data_loaders(
        target_dataset_path,
        batch_size=config['batch_size'],
        num_workers=2,
        target_size=config.get('target_size'),
        augment=config.get('augment', False)
    )
    
    if val_loader is None:
        val_loader = test_loader
    
    num_classes = len(classes)
    
    model = SimpleCNN(
        num_classes=num_classes,
        dropout=config['dropout'],
        use_batchnorm=config['use_batchnorm']
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded pretrained weights")
    except:
        print("Warning: Could not load all weights, using partial loading")
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
    
    if config.get('freeze_features', False):
        print("Freezing feature extraction layers...")
        for param in model.features.parameters():
            param.requires_grad = False
    
    print(f'\nModel architecture:')
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    print('\n' + '='*60)
    print('FINE-TUNING')
    print('='*60)
    
    trainer = Trainer(model, train_loader, val_loader, device, config)
    history = trainer.train()
    
    model_dir = Path(output_dir) / 'models'
    model_path = model_dir / f'{dataset_name}_finetuned.pth'
    trainer.save_model(model_path)
    print(f'\nModel saved to {model_path}')
    
    plot_training_history(
        history,
        save_path=Path(output_dir) / 'plots' / f'{dataset_name}_finetuned_training.png'
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
        save_path=Path(output_dir) / 'plots' / f'{dataset_name}_finetuned_confusion_matrix.png'
    )
    
    return {
        'dataset': dataset_name,
        'model': 'CNN_finetuned',
        'config': str(config),
        'accuracy': f"{metrics['accuracy']:.2f}",
        'precision': f"{metrics['precision']:.2f}",
        'recall': f"{metrics['recall']:.2f}",
        'f1': f"{metrics['f1']:.2f}",
        'best_val_acc': f"{trainer.best_val_acc:.2f}"
    }

def main():
    results = []
    
    print('\n' + '#'*60)
    print('FINE-TUNING: IMAGEBITS → LAND PATCHES')
    print('#'*60)
    
    pretrained_model = 'results/cnn_experiments/models/imagebits_SimpleCNN_optimized_cnn.pth'
    
    if not Path(pretrained_model).exists():
        print(f"Error: Pretrained model not found at {pretrained_model}")
        print("Please train CNN on Imagebits first using: python train_cnn.py")
        return
    
    finetune_configs = [
        {
            'name': 'finetune_all_layers',
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'adam',
            'lr': 0.0005,
            'weight_decay': 1e-4,
            'batch_size': 128,
            'epochs': 15,
            'scheduler': 'plateau',
            'augment': True,
            'target_size': (96, 96),
            'freeze_features': False
        },
        {
            'name': 'finetune_classifier_only',
            'dropout': 0.3,
            'use_batchnorm': True,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 128,
            'epochs': 15,
            'scheduler': 'plateau',
            'augment': True,
            'target_size': (96, 96),
            'freeze_features': True
        }
    ]
    
    for config in finetune_configs:
        print(f'\n\n{"="*60}')
        print(f'Configuration: {config["name"]}')
        print(f'{"="*60}')
        
        result = fine_tune_model(
            pretrained_model,
            'land_patches/land_patches',
            f'land_patches_{config["name"]}',
            config,
            'results/finetune_experiments'
        )
        results.append(result)
    
    save_results_table(results, 'results/finetune_experiments/finetune_results_summary.csv')
    
    print("\n\nComparison:")
    print("Check results/finetune_experiments/finetune_results_summary.csv")
    print("vs results/cnn_experiments/cnn_results_summary.csv (Land Patches baseline)")

if __name__ == '__main__':
    main()
