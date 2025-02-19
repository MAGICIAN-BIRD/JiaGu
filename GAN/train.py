import torch
import csv
from torch.utils.data import DataLoader
from datasets.dataset import OracleBoneDataset
from models.models import Generator, Discriminator, ResNetFeatureExtractor
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from utils.save_images import save_generated_images
from models.transformer import PretrainedTransformerExtractor

# 训练函数
def train(opt, dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, fixed_noise, log_filename, visualizer, output_dir, feature_extractor):
    total_iters = 0
    for epoch in range(opt.epochs):
        for i, (imgs, labels) in enumerate(tqdm(dataloader)):
            # 确保每批次大小一致
            batch_size = imgs.size(0)  # 获取当前批次的实际大小
            real_imgs = imgs.to(opt.device)  # 将图像传送到设备
            real_labels = torch.ones(batch_size, 1, device=opt.device)  # 为每个批次设置真实标签
            fake_labels = torch.zeros(batch_size, 1, device=opt.device)  # 为每个批次设置假标签

            # 使用指定的特征提取器提取特征
            features = feature_extractor(real_imgs)

            # 训练判别器
            optimizer_d.zero_grad()

            # 判别器对真实图像的输出
            outputs = discriminator(real_imgs)
            #print(f"Real labels size: {real_labels.size()}, Discriminator outputs size: {outputs.size()}")  # 输出大小对比
            d_loss_real = criterion(outputs, real_labels)  # 目标和输出一致
            d_loss_real.backward()

            # 生成假图像
            z = torch.randn(batch_size, opt.z_dim, 1, 1, device=opt.device)  # 随机噪声
            fake_imgs = generator(z, features)  # 将特征和噪声输入生成器

            # 判别器对假图像的输出
            outputs = discriminator(fake_imgs.detach())  # 使用detach()来阻止梯度反向传播
            d_loss_fake = criterion(outputs, fake_labels)  # 目标和输出一致
            d_loss_fake.backward()

            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)  # 目标和输出一致
            g_loss.backward()
            optimizer_g.step()

            total_iters += batch_size
            if total_iters % opt.log_freq == 0:
                log_data = {
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'D_loss': d_loss_real.item() + d_loss_fake.item(),
                    'G_loss': g_loss.item(),
                }
                with open(log_filename, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['epoch', 'batch', 'D_loss', 'G_loss'])
                    writer.writeheader() if total_iters == batch_size else None
                    writer.writerow(log_data)
                print(f"Epoch [{epoch + 1}/{opt.epochs}], Batch [{i + 1}/{len(dataloader)}] D_loss: {d_loss_real.item() + d_loss_fake.item()}, G_loss: {g_loss.item()}")

            # 保存模型
            if total_iters % opt.save_freq == 0:
                torch.save(generator.state_dict(), f"generator_epoch_{epoch + 1}.pth")
                torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch + 1}.pth")

            # 可视化生成的图片，仅在使用Visdom时
            if visualizer and total_iters % opt.display_freq == 0:
                visualizer.display_current_results(real_imgs, fake_imgs, epoch)

            # 保存生成的图片
            if total_iters % opt.save_freq == 0:
                save_generated_images(fake_imgs, labels, epoch, output_dir)


# 主训练函数
def main():
    opt = TrainOptions().parse()  # 获取命令行选项

    # 设备配置，判断是否有可用的GPU
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为 ViT 所需的 224x224 尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化
    ])

    # 加载数据集
    dataset = OracleBoneDataset(img_dir=opt.dataroot, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # 初始化模型
    generator = Generator(z_dim=opt.z_dim, feature_dim=256).to(opt.device)
    discriminator = Discriminator().to(opt.device)

    # 根据选项选择使用Transformer还是ResNet
    if opt.use_resnet:
        feature_extractor = ResNetFeatureExtractor().to(opt.device)
    else:
        feature_extractor = PretrainedTransformerExtractor(z_dim=opt.z_dim).to(opt.device)

    # 优化器
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # 损失函数
    criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss，以适应sigmoid输出

    # 日志文件
    log_filename = 'training_log.csv'

    # 固定噪声用于生成图像
    fixed_noise = torch.randn(opt.batch_size, opt.z_dim, 1, 1, device=opt.device)

    # 可视化工具 (如果use_visdom为True)
    visualizer = None
    if opt.use_visdom:
        visualizer = Visualizer(opt)

    # 生成图像的保存路径
    output_dir = 'data/generated_images'

    # 训练
    train(opt, dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, fixed_noise, log_filename, visualizer, output_dir, feature_extractor)


if __name__ == '__main__':
    main()
