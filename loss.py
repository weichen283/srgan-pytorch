import torch
from torch import nn
from torchvision.models.vgg import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features.children())[:12])

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, fake_out, fake, real):

        adversarial_loss = torch.mean(1 - fake_out)

        content_loss = self.mse_loss(fake, real) + 0.006 * self.mse_loss(self.loss_network(fake), self.loss_network(real))

        total_loss = content_loss + 1e-3 * adversarial_loss

        return total_loss


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
