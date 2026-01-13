import os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset_info(dataset_path, split='train'):
    dataset_path = Path(dataset_path) / split
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    class_counts = {}
    image_files = {}
    
    for cls in classes:
        cls_path = dataset_path / cls
        images = list(cls_path.glob('*.png')) + list(cls_path.glob('*.jpg'))
        class_counts[cls] = len(images)
        image_files[cls] = images
    
    return classes, class_counts, image_files

def plot_class_distribution(class_counts, split_name, dataset_name, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax.bar(classes, counts, color='steelblue', alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title(f'{dataset_name} - {split_name} Set Class Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def load_images_from_class(image_files, num_samples=10):
    images = []
    for img_path in image_files[:num_samples]:
        img = Image.open(img_path)
        images.append(np.array(img))
    return images

def plot_sample_images(image_files, classes, dataset_name, samples_per_class=5, save_path=None):
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                             figsize=(samples_per_class * 2, num_classes * 2))
    
    fig.suptitle(f'{dataset_name} - Sample Images per Class', fontsize=16, fontweight='bold')
    
    for i, cls in enumerate(classes):
        images = load_images_from_class(image_files[cls], samples_per_class)
        
        for j, img in enumerate(images):
            ax = axes[i, j] if num_classes > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(cls, fontsize=10, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def compute_image_statistics(image_files, classes):
    stats = {cls: {'mean': [], 'std': [], 'sizes': []} for cls in classes}
    
    for cls in classes:
        for img_path in image_files[cls]:
            img = np.array(Image.open(img_path))
            
            if len(img.shape) == 3:
                stats[cls]['mean'].append(img.mean(axis=(0, 1)))
                stats[cls]['std'].append(img.std(axis=(0, 1)))
            else:
                stats[cls]['mean'].append(img.mean())
                stats[cls]['std'].append(img.std())
            
            stats[cls]['sizes'].append(img.shape[:2])
    
    summary = {}
    for cls in classes:
        summary[cls] = {
            'mean_intensity': np.mean(stats[cls]['mean'], axis=0),
            'std_intensity': np.mean(stats[cls]['std'], axis=0),
            'image_sizes': Counter([tuple(s) for s in stats[cls]['sizes']])
        }
    
    return summary

def plot_intensity_distributions(stats, dataset_name, save_path=None):
    classes = list(stats.keys())
    means = [stats[cls]['mean_intensity'] for cls in classes]
    stds = [stats[cls]['std_intensity'] for cls in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if isinstance(means[0], np.ndarray):
        mean_values = np.array([m.mean() for m in means])
        std_values = np.array([s.mean() for s in stds])
    else:
        mean_values = np.array(means)
        std_values = np.array(stds)
    
    ax1.bar(classes, mean_values, color='coral', alpha=0.8)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Mean Intensity', fontsize=12)
    ax1.set_title('Average Pixel Intensity per Class', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(classes, std_values, color='lightgreen', alpha=0.8)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Std Intensity', fontsize=12)
    ax2.set_title('Pixel Intensity Variability per Class', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
