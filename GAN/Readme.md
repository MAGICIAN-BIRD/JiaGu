OracleBone GAN 项目
该项目实现了一个基于生成对抗网络（GAN）的甲骨文图像生成系统。通过结合特征提取器（如ResNet或Transformer）和生成器/判别器模型，本系统可以生成逼真的甲骨文图像，支持不同类型的特征提取方式。以下是项目的详细介绍，包括各个模块的功能和如何使用该代码。

项目结构
GAN/
 ├── datasets/
 │   └── dataset.py
 ├──data/
 │   └── 
 ├── models/
 │   ├── models.py
 │   └── transformer.py
 ├── options/
 │   ├── base_options.py
 │   └── train_options.py
 ├── utils/
 │   ├── save_images.py
 │   └── visualizer.py
 ├── train.py
 └──training_log.csv

功能说明
1. 数据集加载 (dataset.py)
OracleBoneDataset 是一个自定义的数据集加载类，继承自 torch.utils.data.Dataset。它用于加载甲骨文图像数据集，并且会自动按文件夹结构进行标签的分配。

主要功能：
从指定目录加载图像数据，支持 .png 和 .jpg 格式。
使用文件夹名作为标签，每个文件夹对应一个标签类别。
可以选择是否应用图像预处理（如缩放、标准化）。

2. 生成器与判别器 (models.py)
生成器：该生成器模型使用了一个由 Linear 和 ConvTranspose2d 层组成的深度网络。生成器的输入是噪声向量（潜在变量）和通过特征提取器提取的图像特征，经过处理后生成假图像。

判别器：判别器是一个卷积神经网络（CNN），用于区分真实图像和生成图像。

3. 特征提取器 (transformer.py)
该模块包含基于 ViT（Vision Transformer）和 ResNet 的特征提取器。根据选项的不同，训练时可以选择不同的特征提取器：

ResNet：使用预训练的 ResNet50，去除最后的分类层，仅保留特征提取部分。
Transformer：使用预训练的 ViT 模型进行特征提取。

4. 训练选项 (train_options.py)
该模块定义了训练时使用的各种超参数。你可以根据需求调整如学习率、噪声维度、训练周期数等选项。

主要参数：
z_dim: 噪声向量的维度。
epochs: 训练的总周期数。
use_resnet: 是否使用 ResNet 作为特征提取器。
batch_size: 每次训练时使用的图像数量。

5. 日志记录与可视化 (save_images.py, visualizer.py)
save_images.py：该模块负责将生成的图像保存到文件夹中，并按类别进行存储。每个类别会有一个子文件夹，保存每个训练周期生成的图像。

visualizer.py：用于在训练过程中实时显示生成的图像，通过 Visdom 显示生成的图像和损失值。

6. 训练主脚本 (train.py)
该脚本是整个训练流程的主程序，负责调用数据加载器、模型、优化器、损失函数以及进行模型训练和评估。

主要步骤：
加载数据集。
初始化生成器和判别器。
选择特征提取器（ResNet 或 Transformer）。
定义优化器和损失函数。
启动训练过程，进行迭代更新。
每隔一定轮次保存生成的图像和模型。


安装依赖
确保你的环境中安装了以下库：

torch
torchvision
PIL
visdom (用于可视化)
tqdm (用于显示进度条)

训练流程
准备数据：确保你的数据集文件夹内按类别划分子文件夹，且每个文件夹中包含甲骨文图像文件。

配置训练参数：通过命令行参数配置训练选项。

运行训练：使用以下命令启动训练过程：
python train.py --dataroot data/yout_data_pth --batch_size 64 --n_epochs 100 --z_dim 100 --use_resnet True
