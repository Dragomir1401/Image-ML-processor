import os
import requests
from pathlib import Path

def download_file(url, save_path):
    print(f'Downloading {url}...')
    response = requests.get(url)
    response.raise_for_status()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    
    print(f'Saved to {save_path}')

def download_ro_sent_dataset(output_dir='sentiment_data'):
    base_url = 'https://raw.githubusercontent.com/dumitrescustefan/Romanian-Transformers/examples/examples/sentiment_analysis/ro'
    
    files = {
        'train': f'{base_url}/train.csv',
        'test': f'{base_url}/test.csv'
    }
    
    for split, url in files.items():
        save_path = os.path.join(output_dir, f'{split}.csv')
        download_file(url, save_path)
    
    print(f'\nDataset downloaded successfully to {output_dir}/')

if __name__ == '__main__':
    download_ro_sent_dataset()
