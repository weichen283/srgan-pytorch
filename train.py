
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder
from loss import GeneratorLoss
from model import Generator, Discriminator

if __name__ == '__main__':

    RETRAIN = True
    CROP_SIZE = 32
    UPSCALE_FACTOR = 4
    NUM_EPOCHS = 10
    BATCH_SIZE = 32

    train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

    netG = Generator(UPSCALE_FACTOR)
    print(netG)
    netD = Discriminator()
    print(netD)

    if RETRAIN:
        netG.load_state_dict(torch.load('netG.pth'))
        netD.load_state_dict(torch.load('netD.pth'))
        print("weights loaded")

    generator_criterion = GeneratorLoss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001)

    for epoch in range(1, NUM_EPOCHS + 1):
        mean_generator_loss = 0.0
        mean_discriminator_loss = 0.0
        train_bar = tqdm(train_loader)

        netG.train()
        netD.train()
        for lr, hr in train_bar:

            real_img = Variable(hr)
            fake_img = netG(Variable(lr))
            netD.zero_grad()

            d_loss = 1 - netD(real_img).mean() + netD(fake_img).mean()

            mean_discriminator_loss += d_loss
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            netG.zero_grad()
            g_loss = generator_criterion(netD(fake_img).mean(), fake_img, real_img)
            mean_generator_loss += g_loss
            g_loss.backward()

            optimizerG.step()

            train_bar.desc = '\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f' % (epoch, NUM_EPOCHS, d_loss, g_loss)

        print('\rDiscriminator_Loss: %.4f Generator_Loss: %.4f' % (mean_discriminator_loss / len(train_bar), mean_generator_loss / len(train_bar)))

        # save model parameters
        torch.save(netG.state_dict(), 'netG.pth')
        torch.save(netD.state_dict(), 'netD.pth')
