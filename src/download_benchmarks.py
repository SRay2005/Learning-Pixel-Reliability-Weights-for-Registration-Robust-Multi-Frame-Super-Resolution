"""
Download real benchmark datasets for Super-Resolution:
- Set5 (5 images)
- Set14 (14 images)
- BSD100 (100 images)

Source: jbhuang0604/SelfExSR benchmark repository
"""

import os
import urllib.request
import concurrent.futures
from tqdm import tqdm

DATASETS = {
    'Set5': {
        'count': 5,
        'base_url': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_2/',
        'prefix': 'img_',
        'suffix': '_SRF_2_HR.png'
    },
    'Set14': {
        'count': 14,
        'base_url': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_2/',
        'prefix': 'img_',
        'suffix': '_SRF_2_HR.png'
    },
    'BSD100': {
        'count': 100,
        'base_url': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/BSD100/image_SRF_2/',
        'prefix': 'img_',
        'suffix': '_SRF_2_HR.png'
    }
}


def download_single_image(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return save_path, True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        return save_path, True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return save_path, False


def download_all_benchmarks(target_dir='data/hr'):
    os.makedirs(target_dir, exist_ok=True)
    tasks = []
    
    for dataset_name, meta in DATASETS.items():
        count = meta['count']
        base_url = meta['base_url']
        prefix = meta['prefix']
        suffix = meta['suffix']
        
        for i in range(1, count + 1):
            filename = f"{prefix}{i:03d}{suffix}"
            url = f"{base_url}{filename}"
            save_name = f"{dataset_name.lower()}_{i:03d}.png"
            save_path = os.path.join(target_dir, save_name)
            tasks.append((url, save_path))
            
    print(f"Downloading {len(tasks)} benchmark images into {target_dir}...")
    
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single_image, url, path) for url, path in tasks]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images"):
            _, ok = f.result()
            if ok:
                success += 1
                
    print(f"Successfully prepared {success}/{len(tasks)} real benchmark images in {target_dir}!")
    return success


if __name__ == '__main__':
    download_all_benchmarks()
