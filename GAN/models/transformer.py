import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class PretrainedTransformerExtractor(nn.Module):
    """基于预训练ViT-B/16的特征提取器，包含可调整的分类头部

    Args:
        z_dim (int): 潜在空间维度，默认100
        num_classes (int): 分类任务类别数，默认86
    """

    def __init__(self, z_dim: int = 100, num_classes: int = 86):
        super().__init__()

        # 初始化预训练模型
        self.vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # 模型组件初始化
        self._init_components(z_dim, num_classes)

        # 冻结参数配置
        self._freeze_parameters()

    def _init_components(self, z_dim: int, num_classes: int) -> None:
        """初始化模型组件和参数"""
        # 特征提取部分
        self.patch_embedding = self.vit_model.conv_proj
        self.encoder = self.vit_model.encoder

        # 位置嵌入（包含class token的位置编码）
        self.pos_embedding = self.encoder.pos_embedding

        # 分类头部重构
        in_features = self.vit_model.heads[0].in_features
        self.vit_model.heads = self._create_classifier(in_features, num_classes)

    def _create_classifier(self, in_features: int, num_classes: int) -> nn.Module:
        """创建分类头部"""
        return nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes)
        )

    def _freeze_parameters(self) -> None:
        """冻结不需要训练的参数"""
        # 冻结整个encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 解冻分类头部
        for param in self.vit_model.heads.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为(B, 3, 224, 224)

        Returns:
            torch.Tensor: 分类结果，形状为(B, num_classes)
        """
        # 图像分块嵌入 [B, 3, 224, 224] => [B, 196, 768]
        x = self.patch_embedding(x)  # [B, 768, 14, 14]
        x = x.flatten(2).permute(0, 2, 1)

        # 添加class token [B, 197, 768]
        class_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        x = torch.cat([class_token, x], dim=1)

        # 加入位置编码
        x += self.pos_embedding.to(x.device)

        # 通过Transformer编码器
        x = self.encoder(x)

        # 全局平均池化
        return self.vit_model.heads(x.mean(dim=1))