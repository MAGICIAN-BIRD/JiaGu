import torch
import torch.nn as nn
import torchvision.models as models

class Generator(nn.Module):
    def __init__(self, z_dim, feature_dim):
        super().__init__()

        # 计算 feature_dim, 这里 feature_dim 为 ResNet 输出的维度，即 2048
        self.main = nn.Sequential(
            nn.Linear(z_dim + 2048, 512),  # 确保输入大小正确
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256 * 14 * 14),  # 逐步上采样
            nn.BatchNorm1d(256 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (256, 14, 14)),  # 转为特征图
            # 上采样层
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # [128, 28, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # [64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # [32, 112, 112]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # [3, 224, 224]
            nn.Tanh()
        )

    def forward(self, z, features):
        # 处理噪声输入：将4D噪声张量展平为2D [B, z_dim, 1, 1] => [B, z_dim]
        z_flat = z.view(z.size(0), -1)

        # 处理特征输入：确保特征维度正确
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        # 拼接噪声和特征 [B, z_dim + 2048]
        combined = torch.cat((z_flat, features), dim=1)

        return self.main(combined)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 输入尺寸：[3, 224, 224]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [64, 112, 112]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # [128, 56, 56]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # [256, 28, 28]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),  # 添加全局池化层 [512, 1, 1]
            nn.Flatten(),  # 展平为 [batch_size, 512]
            nn.Linear(512, 1)  # 最终输出 [batch_size, 1]
        )

    def forward(self, x):
        return self.model(x)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        # 使用预训练的 ResNet50
        self.resnet = models.resnet50(pretrained=True)
        # 去掉最后的分类层，保留特征提取部分
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        # 输入形状 [B, 3, 224, 224]
        features = self.resnet(x)  # 输出 [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # 展平为 [B, 2048]
        return features
