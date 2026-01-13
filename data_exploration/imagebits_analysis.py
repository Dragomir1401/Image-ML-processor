import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .utils import (
    load_dataset_info,
    plot_class_distribution,
    plot_sample_images,
    compute_image_statistics,
    plot_intensity_distributions
)

def analyze_imagebits(dataset_path='imagebits/imagebits', output_dir='results/imagebits_analysis'):
    print("=" * 60)
    print("IMAGEBITS DATASET ANALYSIS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        classes, class_counts, image_files = load_dataset_info(dataset_path, split)
        
        print(f"Number of classes: {len(classes)}")
        print(f"Classes: {classes}")
        print(f"Total images: {sum(class_counts.values())}")
        
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} images")
        
        plot_class_distribution(
            class_counts, 
            split.capitalize(), 
            'Imagebits',
            save_path=os.path.join(output_dir, f'class_distribution_{split}.png')
        )
        
        if split == 'train':
            plot_sample_images(
                image_files,
                classes,
                'Imagebits',
                samples_per_class=8,
                save_path=os.path.join(output_dir, 'sample_images.png')
            )
            
            print("\nComputing image statistics...")
            stats = compute_image_statistics(image_files, classes)
            
            print("\nImage statistics per class:")
            for cls in classes:
                print(f"  {cls}:")
                print(f"    Mean intensity: {stats[cls]['mean_intensity']}")
                print(f"    Std intensity: {stats[cls]['std_intensity']}")
                print(f"    Image sizes: {stats[cls]['image_sizes']}")
            
            plot_intensity_distributions(
                stats,
                'Imagebits',
                save_path=os.path.join(output_dir, 'intensity_distributions.png')
            )
    
    print("\n" + "=" * 60)
    print("Analysis complete. Results saved in:", output_dir)
    print("=" * 60)

if __name__ == '__main__':
    analyze_imagebits()
