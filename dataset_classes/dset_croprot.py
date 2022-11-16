import torch
import torch.utils.data
import numpy as np
from dataset_classes.dset_nolabel import DsetNoLabel
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1,2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1,2).flip(2)

class DsetSSCropRot(torch.utils.data.Dataset):
    # Make sure that your dataset only returns one element!
    def __init__(self, dset: DsetNoLabel = None, crop_size: int = 256):
        assert type(dset) is DsetNoLabel, "DsetSSCropRot must take a dataset of type dset_nolabel as input"

        self.dset = dset
        self.crop_size = crop_size

        # use this code to create a 'deterministic' dataset
        # image, _, _, _ = self.dset[0]
        # img_width = image.shape[2]
        # img_height = image.shape[1]

        # self.x_rand = np.random.randint(img_width - self.crop_size, size=len(dset))
        # self.y_rand = np.random.randint(img_height - self.crop_size, size=len(dset))
        # self.labels = np.random.randint(4, size=len(dset))


    def __getitem__(self, index):
        image = self.dset[index]
        if not type(image) is torch.tensor:
            image = torch.tensor(image, dtype=torch.float)

        # randomly make the image black... I used this to test the uncertainty estimates
        # random_black = np.random.randint(2)
        # if random_black == 1:
        #     image = torch.zeros_like(image)
        label = np.random.randint(4)

        img_width = image.shape[2]
        img_height = image.shape[1]

        patch_size = self.crop_size
        # y_zone = 128 + 64

        # generate random position of square patch
        x_rand = np.random.randint(img_width - patch_size)
        # y_rand = np.random.randint(img_height - patch_size - y_zone)
        y_rand = np.random.randint(img_height - patch_size)

        # crop selected patch
        # image = image[:, 128 + y_rand:128 + y_rand+patch_size, x_rand:x_rand+patch_size]
        image = image[:, y_rand:y_rand+patch_size, x_rand:x_rand+patch_size]

        # deterministic dataset
        # x_rand = self.x_rand[index]
        # y_rand = self.y_rand[index]
        # image = image[:, y_rand:y_rand+self.crop_size, x_rand:x_rand+self.crop_size]
        # label = self.labels[index]

        if label == 1:
            image = tensor_rot_90(image)
        elif label == 2:
            image = tensor_rot_180(image)
        elif label == 3:
            image = tensor_rot_270(image)

        return image.numpy(), label

    def __len__(self):
        return len(self.dset)
