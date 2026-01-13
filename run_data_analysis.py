from data_exploration import analyze_imagebits, analyze_land_patches

def main():
    print("Starting data exploration...\n")
    
    print("1. Analyzing Imagebits dataset...")
    analyze_imagebits(
        dataset_path='imagebits/imagebits',
        output_dir='results/imagebits_analysis'
    )
    
    print("\n" + "=" * 60 + "\n")
    
    print("2. Analyzing Land Patches dataset...")
    analyze_land_patches(
        dataset_path='land_patches/land_patches',
        output_dir='results/land_patches_analysis'
    )
    
    print("\n\nAll analyses complete!")

if __name__ == '__main__':
    main()
