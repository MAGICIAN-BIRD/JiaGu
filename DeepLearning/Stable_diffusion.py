import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import multiprocessing
import torch.nn.functional as F
from math import sqrt
import random

# 启用cudnn加速
torch.backends.cudnn.benchmark = True

# 1. 优化后的数据加载部分
class ImageFolderWithPaths(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 使用列表推导式提高文件遍历速度
        self.img_paths = [os.path.join(root, file)
                          for root, dirs, files in os.walk(root_dir)
                          for file in files
                          if file.lower().endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # 返回路径以便后续使用


# 使用更高效的数据增强
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolderWithPaths(root_dir='need_moben', transform=transform)
# 增大batch_size并优化num_workers
dataloader = DataLoader(dataset, batch_size=16, shuffle=True,
                        num_workers=multiprocessing.cpu_count(), pin_memory=True)


# 2. 改进的U-Net结构（加入残差连接和注意力机制）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, timesteps=100):
        super().__init__()
        self.time_embed = nn.Embedding(timesteps, 128)

        # 下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 64x64
        )

        # 中间层
        self.mid = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 32x32
            ResidualBlock(256),
        )

        # 上采样
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t)[:, :, None, None]
        x = self.down1(x) + t_emb
        x = self.mid(x)
        x = self.up1(x)
        return x


# 3. 向量化扩散过程
class Diffusion:
    def __init__(self, timesteps=100, device='cuda'):
        self.timesteps = timesteps
        self.device = device  # 设置设备

        # 线性噪声调度
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=self.device)  # 使用设备
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)  # 确保在设备上
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod).to(self.device)  # 使用设备
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod).to(self.device)  # 使用设备

    def add_noise(self, x_0, t):
        # 确保`t`也在相同的设备上（GPU）
        t = t.to(self.device)  # 将`t`移动到与`sqrt_alpha_cumprod`相同的设备

        # 向量化噪声添加
        noise = torch.randn_like(x_0, device=self.device)
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    # 生成图像的方法
    # 生成图像的方法
    @torch.no_grad()
    def sample(self, model, n_samples=1):
        model.eval()
        x = torch.randn((n_samples, 3, 128, 128), device=self.device)  # 使用设备

        # 使用字符串"cuda"或"cpu"作为device_type
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n_samples,), t, device=self.device)
            with torch.amp.autocast(device_type=str(self.device).split(':')[0]):  # 确保是"cuda"或"cpu"
                pred_noise = model(x, t_batch)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x, device=self.device)  # 使用设备
            else:
                noise = 0

            x = (x - beta_t * pred_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
            x += torch.sqrt(beta_t) * noise

        return x.clamp(-1, 1)


# 4. 混合精度训练
scaler = torch.amp.GradScaler()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    diffusion = Diffusion(device=device)  # 传递设备给Diffusion类

    epochs = 10
    timesteps = diffusion.timesteps

    # 5. 定义每个epoch保存示例图像的逻辑
    os.makedirs('try', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", ncols=100)

        for images, img_paths in progress_bar:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)

            # 随机采样时间步
            t = torch.randint(0, timesteps, (batch_size,), device=device)

            with torch.amp.autocast(device_type='cuda'):  # 修正为明确指定设备类型
                # 添加噪声并预测噪声
                x_t, noise = diffusion.add_noise(images, t)
                predicted_noise = model(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)

            # 混合精度反向传播
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(loss=loss.item())

        # 每一轮随机挑选几张图保存
        if (epoch + 1) % 5 == 0:  # 每5轮保存一次
            random_idx = random.sample(range(batch_size), 3)  # 随机选取3张图片
            selected_images = images[random_idx]
            save_image((selected_images + 1) / 2, f'try/epoch_{epoch+1}_sample.png')

    # 6. 训练完成后，生成每种类型的图片
    @torch.no_grad()
    def generate_and_save_images(model, class_names, n_samples=2):
        model.eval()
        for class_name in class_names:
            for i in range(n_samples):
                x = torch.randn((1, 3, 128, 128), device=device)
                generated_image = diffusion.sample(model, n_samples=1)
                save_image((generated_image + 1) / 2, f'try/{class_name}_{i+1}.png')

    # 获取所有类别名称
    class_names = [os.path.basename(os.path.dirname(path)) for path in dataset.img_paths]
    class_names = list(set(class_names))  # 去重

    # 生成每种类型的图片
    generate_and_save_images(model, class_names)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    train()
    print("Finished!")
