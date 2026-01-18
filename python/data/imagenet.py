"""
ImageNet / Tiny-ImageNet 数据加载器
"""
import os
import numpy as np
from PIL import Image

class TinyImageNetDataset:
    """
    Tiny-ImageNet 数据集 (200类, 64x64)
    下载地址: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    def __init__(self, root='./data/tiny-imagenet-200', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        if train:
            self.data_dir = os.path.join(root, 'train')
        else:
            self.data_dir = os.path.join(root, 'val')
        
        self.samples = []
        self.class_to_idx = {}
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} {'train' if train else 'val'} samples, {len(self.class_to_idx)} classes")
    
    def _load_samples(self):
        if self.train:
            self._load_train()
        else:
            self._load_val()
    
    def _load_train(self):
        """加载训练集: train/n01443537/images/*.JPEG"""
        classes = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))])
        
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            img_dir = os.path.join(self.data_dir, cls, 'images')
            if os.path.exists(img_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.endswith('.JPEG'):
                        self.samples.append((os.path.join(img_dir, img_name), idx))
    
    def _load_val(self):
        """加载验证集: val/images/*.JPEG + val_annotations.txt"""
        # 加载类别映射
        train_dir = os.path.join(self.root, 'train')
        classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
        
        # 加载验证集标注
        val_annotations = os.path.join(self.data_dir, 'val_annotations.txt')
        img_to_class = {}
        if os.path.exists(val_annotations):
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    img_to_class[img_name] = class_name
        
        img_dir = os.path.join(self.data_dir, 'images')
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.JPEG') and img_name in img_to_class:
                    cls = img_to_class[img_name]
                    if cls in self.class_to_idx:
                        self.samples.append((os.path.join(img_dir, img_name), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            # 默认转换
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        
        return img.astype(np.float32), label


class ImageNet100Dataset:
    """
    ImageNet-100 子集 (100类, 224x224)
    需要自己从 ImageNet 提取或下载预制版本
    目录结构: imagenet100/train/n01440764/*.JPEG
    """
    def __init__(self, root='./data/imagenet100', train=True, img_size=224):
        self.root = root
        self.train = train
        self.img_size = img_size
        
        split = 'train' if train else 'val'
        self.data_dir = os.path.join(root, split)
        
        self.samples = []
        self.class_to_idx = {}
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples, {len(self.class_to_idx)} classes")
    
    def _load_samples(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")
        
        classes = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))])
        
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append((os.path.join(cls_dir, img_name), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Resize and center crop
        img = self._resize_crop(img, self.img_size)
        
        # Normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        
        return img.astype(np.float32), label
    
    def _resize_crop(self, img, size):
        """Resize并中心裁剪"""
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Center crop
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))
        return img


def download_tiny_imagenet(root='./data'):
    """下载 Tiny-ImageNet"""
    import urllib.request
    import zipfile
    
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
    extract_dir = os.path.join(root, 'tiny-imagenet-200')
    
    if os.path.exists(extract_dir):
        print(f"Tiny-ImageNet already exists at {extract_dir}")
        return extract_dir
    
    os.makedirs(root, exist_ok=True)
    
    print(f"Downloading Tiny-ImageNet from {url}...")
    print("This may take a while (~240MB)...")
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    os.remove(zip_path)
    print(f"Done! Dataset saved to {extract_dir}")
    
    return extract_dir
