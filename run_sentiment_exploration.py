from sentiment_exploration import analyze_sentiment_data
import os

def main():
    if not os.path.exists('sentiment_data/train.csv') or not os.path.exists('sentiment_data/test.csv'):
        print("Dataset not found!")
        print("Please run first: python download_sentiment_data.py")
        return
    
    print("Analyzing sentiment dataset...")
    analyze_sentiment_data()

if __name__ == '__main__':
    main()
