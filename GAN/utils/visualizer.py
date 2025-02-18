import visdom
import torchvision.utils as vutils

class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.vis = visdom.Visdom(port=opt.display_port, env=opt.display_env, server=opt.display_server)

    def display_current_results(self, real_imgs, fake_imgs, epoch):
        real_imgs = real_imgs.cpu().detach()
        fake_imgs = fake_imgs.cpu().detach()

        real_imgs_grid = vutils.make_grid(real_imgs, padding=2, normalize=True)
        fake_imgs_grid = vutils.make_grid(fake_imgs, padding=2, normalize=True)

        self.vis.image(real_imgs_grid, win="Real Images", opts=dict(title=f"Epoch {epoch} Real Images"))
        self.vis.image(fake_imgs_grid, win="Fake Images", opts=dict(title=f"Epoch {epoch} Fake Images"))

    def print_current_losses(self, epoch, iter, losses):
        self.vis.text(f"Epoch {epoch} Iter {iter} - Losses: {losses}")
