import numpy as np

class CIFAR10Dataset:
    def __init__(self, root='./data', train=True, download=True):
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
 
        self.data = np.stack([ds[i][0].numpy() for i in range(len(ds))], dtype=np.float32)
        self.targets = np.array([ds[i][1] for i in range(len(ds))], dtype=np.float32)
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

class DataLoader:
    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx) - self.batch_size + 1, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            # 支持两种模式：预加载(data/targets)和懒加载(__getitem__)
            if hasattr(self.dataset, 'data') and hasattr(self.dataset, 'targets'):
                yield self.dataset.data[batch_idx], self.dataset.targets[batch_idx]
            else:
                # 懒加载模式：逐个获取并堆叠
                batch_data = []
                batch_labels = []
                for j in batch_idx:
                    data, label = self.dataset[j]
                    batch_data.append(data)
                    batch_labels.append(label)
                yield np.stack(batch_data), np.array(batch_labels, dtype=np.float32)