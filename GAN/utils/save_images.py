import os
import torchvision.utils as vutils


def save_generated_images(fake_imgs, labels, epoch, output_dir, n_images=10):
    """
    保存生成的图像到指定的类别子文件夹中
    :param fake_imgs: 生成的图片
    :param labels: 图片的标签
    :param epoch: 当前训练轮次
    :param output_dir: 输出的根目录
    :param n_images: 每个类别保存的图片数量
    """
    fake_imgs = fake_imgs.cpu()

    for i in range(n_images):
        # 获取标签对应的类别文件夹路径
        label = labels[i].item()
        class_folder = os.path.join(output_dir, str(label))
        os.makedirs(class_folder, exist_ok=True)

        # 保存图片
        img_filename = os.path.join(class_folder, f"epoch_{epoch}_img_{i}.png")
        vutils.save_image(fake_imgs[i], img_filename, normalize=True)
