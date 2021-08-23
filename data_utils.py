from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def display_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(400),
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return self.normalize(lr_image), self.normalize(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = transforms.Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = transforms.Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = transforms.CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_restore_img), transforms.ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = transforms.Resize((self.upscale_factor * h, self.upscale_factor * w),
                                     interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, transforms.ToTensor()(lr_image), transforms.ToTensor()(
            hr_restore_img), transforms.ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
