from os.path import join
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import ValDatasetFromFolder, display_transform
from model import Generator


val_set = ValDatasetFromFolder('data/flowers_dataset/flower_data/val', upscale_factor=4)
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

model = Generator(4).eval()

model.load_state_dict(torch.load('netG.pth'))

print("weights loaded")

out_path = './results'
with torch.no_grad():
    val_bar = tqdm(val_loader, desc='[Image processing...]')
    val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        lr = val_lr
        hr = val_hr
        sr = model(lr)
        val_images.extend([display_transform()(val_hr_restore.squeeze(0)),
                           display_transform()(sr.data.cpu().squeeze(0)),
                           display_transform()(hr.data.cpu().squeeze(0))])

    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 3)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, join(out_path, 'index%d.png') % index, padding=5)
        index += 1
