import os
import io
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(EMOTION_CLASSES)}

PARQUET_URLS = {
    'train': 'https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/train-00000-of-00001-5eab84e1c6a2fc27.parquet',
    'publicTest': 'https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet',
    'privateTest': 'https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet',
}

def download_file(url: str, dest_path: str):
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
    print(f"Downloaded {dest_path} successfully ({os.path.getsize(dest_path) / (1024*1024):.2f} MB).")
    return dest_path

def prepare_data_cache(data_dir: str = 'data'):
    """Download parquets and prepare fast cached .npz files for train, val, and test."""
    os.makedirs(data_dir, exist_ok=True)
    cache_files = {
        'train': os.path.join(data_dir, 'fer2013_train.npz'),
        'val': os.path.join(data_dir, 'fer2013_val.npz'),      # publicTest
        'test': os.path.join(data_dir, 'fer2013_test.npz'),    # privateTest
    }

    all_exist = all(os.path.exists(p) for p in cache_files.values())
    if all_exist:
        print("Cached npz files already exist.")
        return cache_files

    split_map = {'train': 'train', 'val': 'publicTest', 'test': 'privateTest'}
    for split, parquet_key in split_map.items():
        cache_path = cache_files[split]
        if os.path.exists(cache_path):
            continue

        parquet_path = os.path.join(data_dir, f"{parquet_key}.parquet")
        download_file(PARQUET_URLS[parquet_key], parquet_path)
        
        print(f"Parsing {parquet_path} into cached array...")
        df = pd.read_parquet(parquet_path)
        labels = df['label'].to_numpy(dtype=np.int64)
        
        # Decode images into uint8 array (N, 48, 48, 3)
        num_samples = len(df)
        images = np.empty((num_samples, 48, 48, 3), dtype=np.uint8)
        for i, img_dict in enumerate(df['image']):
            img_bytes = img_dict['bytes'] if isinstance(img_dict, dict) else img_dict
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            images[i] = np.array(pil_img, dtype=np.uint8)

        np.savez_compressed(cache_path, images=images, labels=labels)
        print(f"Saved {cache_path} with {num_samples} samples.")

    return cache_files

class FER2013Dataset(Dataset):
    """FER2013 PyTorch Dataset supporting data augmentations and cached arrays."""
    def __init__(self, npz_path: str, transform=None):
        data = np.load(npz_path)
        self.images = data['images']  # shape: (N, 48, 48, 3), uint8
        self.labels = data['labels']  # shape: (N,), int64
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        pil_img = Image.fromarray(img_np)
        label = int(self.labels[idx])

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = T.ToTensor()(pil_img)

        return tensor_img, label

def get_transforms(img_size: int = 112):
    """
    Returns train and validation transforms.
    Train includes: RandomHorizontalFlip, RandomRotation, ColorJitter, Affine, Normalize.
    Val includes: Resize, Normalize.
    """
    # Standard ImageNet normalization stats
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomCrop(img_size, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, val_transform

def compute_class_weights(labels: np.ndarray, beta: float = 0.5) -> torch.Tensor:
    """
    Compute smoothed inverse frequency class weights to address class imbalance.
    weight_c = (total / (num_classes * count_c)) ^ beta
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    num_classes = len(classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        weights[c] = (total / (num_classes * cnt)) ** beta
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(data_dir: str = 'data', img_size: int = 112, batch_size: int = 64, num_workers: int = 0):
    """Prepares cache and returns train, val, test dataloaders and class weights."""
    cache_files = prepare_data_cache(data_dir)
    train_tf, val_tf = get_transforms(img_size=img_size)

    train_ds = FER2013Dataset(cache_files['train'], transform=train_tf)
    val_ds = FER2013Dataset(cache_files['val'], transform=val_tf)
    test_ds = FER2013Dataset(cache_files['test'], transform=val_tf)

    class_weights = compute_class_weights(train_ds.labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights

def verify_dataset():
    """Verify loop: load a batch, print shapes, save a sample grid, confirm labels."""
    print("--- Starting Dataset Verification Loop ---")
    import matplotlib.pyplot as plt

    train_loader, val_loader, test_loader, weights = get_dataloaders(img_size=112, batch_size=16)
    
    print(f"Train batches: {len(train_loader)} (total {len(train_loader.dataset)} samples)")
    print(f"Val batches:   {len(val_loader)} (total {len(val_loader.dataset)} samples)")
    print(f"Test batches:  {len(test_loader)} (total {len(test_loader.dataset)} samples)")
    print(f"Class weights: {weights.tolist()}")

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch images shape: {batch_imgs.shape}, dtype: {batch_imgs.dtype}")
    print(f"Batch labels shape: {batch_labels.shape}, dtype: {batch_labels.dtype}")
    print(f"Tensor min: {batch_imgs.min().item():.3f}, max: {batch_imgs.max().item():.3f}")

    # Denormalize for plotting
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm_imgs = batch_imgs * norm_std + norm_mean
    denorm_imgs = torch.clamp(denorm_imgs, 0.0, 1.0)

    # Save 4x4 grid of sample batch
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img_np = denorm_imgs[i].permute(1, 2, 0).numpy()
        lbl_idx = batch_labels[i].item()
        emotion = EMOTION_CLASSES[lbl_idx]
        ax.imshow(img_np)
        ax.set_title(f"{emotion} ({lbl_idx})", fontsize=10)
        ax.axis('off')

    os.makedirs('reports', exist_ok=True)
    grid_path = os.path.join('reports', 'sample_batch.png')
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Sample batch grid saved to {grid_path}")
    print("--- Dataset Verification PASSED ---")

if __name__ == '__main__':
    verify_dataset()
