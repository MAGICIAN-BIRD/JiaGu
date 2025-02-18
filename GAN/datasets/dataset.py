import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OracleBoneDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        甲骨文数据集加载类

        :param img_dir: 数据集文件夹路径
        :param transform: 图像转换操作
        """
        self.img_dir = img_dir
        self.img_paths = []
        self.labels = []
        self.label_map = {}

        # 遍历子文件夹（每个子文件夹代表一个类别标签）
        for label_id, folder_name in enumerate(os.listdir(img_dir)):
            folder_path = os.path.join(img_dir, folder_name)
            if os.path.isdir(folder_path):
                self.label_map[label_id] = folder_name
                for fname in os.listdir(folder_path):
                    if fname.endswith('.png') or fname.endswith('.jpg'):
                        self.img_paths.append(os.path.join(folder_path, fname))
                        self.labels.append(label_id)

        self.transform = transform

    def __len__(self):
        """返回数据集中的图像数量"""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """加载并转换图像"""
        img = Image.open(self.img_paths[idx]).convert('RGB')  # 读取图像并转换为RGB格式
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
