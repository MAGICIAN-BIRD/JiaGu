import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np
import random
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.nn.init as init

# 1. 数据准备：获取类标签
root_dir = r'Y:\py-torch\甲骨文切割\need_moben'  # 你的数据集路径
classes = os.listdir(root_dir)  # 获取子文件夹（类标签）

# 2. 数据加载与预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 统一图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 将图像标准化到[-1, 1]范围
])

# 加载数据集
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


# 3. 定义权重初始化函数
def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0.0, 0.02)  # 正态分布初始化
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


# 4. 定义DCGAN模型（生成器和判别器）

# 生成器
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_map_size=64):
        super(DCGANGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围为[-1, 1]，与数据集预处理一致
        )

    def forward(self, x):
        return self.generator(x)


# 判别器
class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map_size=64):
        super(DCGANDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出概率值
        )

    def forward(self, x):
        return self.discriminator(x)


# 5. 损失函数和优化器
def wgan_loss(output, real=True):
    if real:
        return -torch.mean(output)
    else:
        return torch.mean(output)


z_dim = 100  # 噪声向量的维度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = DCGANGenerator(z_dim=z_dim).to(device)
discriminator = DCGANDiscriminator().to(device)

generator.apply(weights_init)  # 初始化生成器
discriminator.apply(weights_init)  # 初始化判别器

optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.5)

# 6. 训练DCGAN模型
epochs = 30  # 增加训练周期


def train(rank, generator, discriminator, dataloader, device):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # 创建标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 训练判别器：最大化 D(x) - D(G(z))
            optimizer_d.zero_grad()

            # 真实图像
            outputs = discriminator(real_images)
            d_loss_real = wgan_loss(outputs, real=True)
            d_loss_real.backward()

            # 生成的假图像
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = wgan_loss(outputs, real=False)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()

            # 训练生成器：最大化 D(G(z))
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = wgan_loss(outputs, real=True)
            g_loss.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # 每个epoch保存生成的图像
        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 10 == 0:
            output_dir = f'Y:/py-torch/甲骨文切割/generated_images_epoch_{epoch + 1}'
            os.makedirs(output_dir, exist_ok=True)
            z = torch.randn(64, z_dim, 1, 1).to(device)
            fake_images = generator(z)
            save_image(fake_images, os.path.join(output_dir, f"fake_images_epoch_{epoch + 1}.png"), nrow=8,
                       normalize=True)
    print(f"Training complete for process {rank}!")


# 进程保护
if __name__ == '__main__':
    mp.spawn(train, nprocs=4, args=(generator, discriminator, dataloader, device))  # 使用4个进程训练


# 7. 生成图像并按类保存
output_dir = r'Y:\py-torch\甲骨文切割\GAN_test'  # 生成图片保存的路径
os.makedirs(output_dir, exist_ok=True)

num_images_per_class = 10  # 每个类生成10张图片

for class_name in classes:
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # 生成对应类的图像
    for i in tqdm(range(num_images_per_class), desc=f"Generating images for {class_name}"):
        # 生成随机噪声
        z = torch.randn(1, z_dim, 1, 1, device=device)

        # 生成图片
        with torch.no_grad():
            generated_image = generator(z)

        # 将生成的图片保存到对应的文件夹
        image_filename = os.path.join(class_dir, f'{class_name}_{i}.png')
        save_image(generated_image, image_filename, normalize=True)

print("Image generation complete!")
