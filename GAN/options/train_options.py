from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    """继承BaseOptions的训练选项类"""

    def initialize(self, parser):
        # 先添加父类参数
        parser = super().initialize(parser)
        # 添加训练专用参数
        parser.add_argument('--z_dim', type=int, default=100, help='噪声维度')
        parser.add_argument('--epochs', type=int, default=100, help='训练周期数')
        parser.add_argument('--log_freq', type=int, default=100, help='日志记录频率')
        parser.add_argument('--display_freq', type=int, default=40, help='图像显示频率')

        # 添加 ResNet 是否启用参数
        parser.add_argument('--use_resnet', type=bool, default=False, help='是否使用ResNet作为特征提取器')

        # 添加是否启用 Visdom 参数
        parser.add_argument('--use_visdom', type=bool, default=False, help='是否使用Visdom进行可视化')

        # 添加可视化参数
        parser.add_argument('--display_port', type=int, default=8097, help='Visdom端口')
        parser.add_argument('--display_env', type=str, default='main', help='Visdom环境')
        parser.add_argument('--display_server', type=str, default='http://localhost', help='Visdom服务器地址')

        self.isTrain = True
        return parser
