import argparse

class BaseOptions:
    """This class定义了所有模型的公共选项。"""

    def initialize(self, parser):
        """
        初始化公共选项
        """
        # 添加公共参数到传入的解析器
        parser.add_argument('--dataroot', type=str, default='./datasets', help='数据集根目录')
        parser.add_argument('--batch_size', type=int, default=64, help='每批图像数量')
        parser.add_argument('--n_epochs', type=int, default=100, help='训练的总周期数')
        parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1参数')
        parser.add_argument('--save_freq', type=int, default=5000, help='模型保存频率')
        return parser

    def parse(self):
        """解析命令行参数。"""
        # 创建解析器并添加参数
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize(parser)
        return parser.parse_args()