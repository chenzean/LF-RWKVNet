import os

import cv2
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
import random
from utils import *
from einops import rearrange
from utils.imresize import imresize


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.patch_size = args.patch_size
        self.scale_factor = args.scale_factor
        self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                           str(args.scale_factor) + 'x/'
        self.ref_dataset_dir = args.path_for_train + 'ref_' + str(args.scale_factor) + 'x/'

        self.data_list = os.listdir(self.dataset_dir)
        self.ref_list = os.listdir(self.ref_dataset_dir)

        self.file_list = []
        self.ref_file_list = []

        # input lr image
        for index, _ in enumerate(self.data_list):
            self.file_list.extend([self.data_list[index]])

        #  stack of 2D HR ref images
        for index, _ in enumerate(self.ref_list):
            self.ref_file_list.extend([self.ref_list[index]])

        # Determine the length of the dataset
        if len(self.ref_file_list) == len(self.file_list):
            self.item_num = len(self.file_list)
        else:
            print('There are problems with the dataset. (Different number of reference and low-resolution images)')

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        ref_file_name = [self.ref_dataset_dir + self.ref_file_list[index]]
        Lr_angRes_in = self.angRes
        Lr_angRes_out = self.angRes

        # input LR LF images and 2D HR reference images
        with h5py.File(file_name[0], 'r') as hf1, h5py.File(ref_file_name[0], 'r') as hf2:
            Lr_SAI_y = np.array(hf1.get('Lr_SAI_y')).astype(np.float32) # Lr_SAI_y      60 60
            Hr_SAI_y = np.array(hf1.get('Hr_SAI_y')).astype(np.float32) # Hr_SAI_y      480 480
            ref_SAI_y = np.array(hf2.get('ref_sai')).astype(np.float32)  # ref_sai_y
            # random_number = random.randint(0, 55)
            # print(ref_SAI_y.shape)        # (128, 128, 56)
            # ref_y = ref_SAI_y[:,:,random_number]
            ref_y = ref_SAI_y[:, :]

            hr = rearrange(Hr_SAI_y, '(an1 h) (an2 w)->h w (an1 an2)', an1=self.angRes, an2=self.angRes,h=self.patch_size * self.scale_factor, w=self.patch_size * self.scale_factor)
            hr_down4 = imresize(hr, 1 / 4, method='bicubic')
            hr_down2 = imresize(hr, 1 / 2, method='bicubic')
            hr_down4 = rearrange(hr_down4, 'h w (an1 an2)->(an1 h) (an2 w)',an1=self.angRes, an2=self.angRes, h=self.patch_size * 2, w=self.patch_size * 2)
            hr_down2 = rearrange(hr_down2, 'h w (an1 an2)->(an1 h) (an2 w)',an1=self.angRes, an2=self.angRes, h=self.patch_size *4, w=self.patch_size *4)

            data, label, ref, label_down4, label_down2 = augmentation_8(Lr_SAI_y, Hr_SAI_y, ref_y, hr_down4, hr_down2)

            Lr_SAI_y = ToTensor()(data.copy())
            Hr_SAI_y = ToTensor()(label.copy())
            ref_SAI_y = ToTensor()(ref.copy())
            hr_down4 = ToTensor()(label_down4.copy())
            hr_down2 = ToTensor()(label_down2.copy())

            return Lr_SAI_y, Hr_SAI_y, ref_SAI_y, hr_down4, hr_down2, [Lr_angRes_in, Lr_angRes_out]








    def __len__(self):
        return self.item_num

# class ValSetDataLoader(Dataset):
#     def __init__(self, args):
#         super(ValSetDataLoader, self).__init__()
#         self.angRes = args.angRes
#         self.dataset_dir = args.path_for_val + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
#                            str(args.scale_factor) + 'x/'
#         self.ref_dataset_dir = args.path_for_val + 'ref_' + str(args.scale_factor) + 'x/'
#
#         self.data_list = os.listdir(self.dataset_dir)
#         self.ref_list = os.listdir(self.ref_dataset_dir)
#
#         self.file_list = []
#         self.ref_file_list = []
#
#         # input lr image
#         for index, _ in enumerate(self.data_list):
#             self.file_list.extend([self.data_list[index]])
#
#         #  stack of 2D HR ref images
#         for index, _ in enumerate(self.ref_list):
#             self.ref_file_list.extend([self.ref_list[index]])
#
#         # Determine the length of the dataset
#         if len(self.ref_file_list) == len(self.file_list):
#             self.item_num = len(self.file_list)
#         else:
#             print('There are problems with the dataset. (Different number of reference and low-resolution images)')
#
#     def __getitem__(self, index):
#         file_name = [self.dataset_dir + self.file_list[index]]
#         ref_file_name = [self.ref_dataset_dir + self.ref_file_list[index]]
#         Lr_angRes_in = self.angRes
#         Lr_angRes_out = self.angRes
#
#         # input LR LF images and 2D HR reference images
#         with h5py.File(file_name[0], 'r') as hf1, h5py.File(ref_file_name[0], 'r') as hf2:
#             Lr_SAI_y = np.array(hf1.get('Lr_SAI_y')).astype(np.float32) # Lr_SAI_y
#             Hr_SAI_y = np.array(hf1.get('Hr_SAI_y')).astype(np.float32) # Hr_SAI_y
#             ref_SAI_y = np.array(hf2.get('ref_sai')).astype(np.float32)  # ref_sai_y
#             # random_number = random.randint(0, 55)
#             # print(ref_SAI_y.shape)        # (128, 128, 56)
#             # ref_y = ref_SAI_y[:,:,random_number]
#             ref_y = ref_SAI_y[:, :]
#
#             Lr_SAI_y, Hr_SAI_y, ref_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y, ref_y)
#             Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
#             Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
#             ref_SAI_y = ToTensor()(ref_SAI_y.copy())
#             LF_name = self.file_list[index].split('/')[-1].split('.')[0]
#
#         return Lr_SAI_y, Hr_SAI_y, ref_SAI_y, [Lr_angRes_in, Lr_angRes_out], LF_name
#
#     def __len__(self):
#         return self.item_num

def MultiValSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    dataset_dir = args.path_for_val + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                      str(args.scale_factor) + 'x/'
    data_list = os.listdir(dataset_dir)


    test_Loaders = []


    val_Dataset = ValSetDataLoader(args)
    length_of_vals = len(val_Dataset)


    val_Loaders = DataLoader(dataset=val_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    return data_list, val_Loaders, length_of_vals

class ValSetDataLoader(Dataset):
    def __init__(self, args, Lr_Info=None):
        super(ValSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.dataset_dir = args.path_for_val + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                           str(args.scale_factor) + 'x/'
        self.ref_dataset_dir = args.path_for_val + 'ref_' + str(args.scale_factor) + 'x/'

        self.dataset_list = os.listdir(self.dataset_dir)
        self.ref_list = os.listdir(self.ref_dataset_dir)

        self.file_list = []
        self.ref_file_list = []

        # self.random_number = random.randint(0, 55)
        # The 'random_number'st in the 2D HR image stack is taken out as the 2D reference image
        # print(f"The '{self.random_number}'st in the 2D HR image stack is taken out as the 2D reference image")

        # input LR LF images
        for index, _ in enumerate(self.dataset_list):
            self.file_list.extend([self.dataset_list[index]])

        #  stack of 2D HR ref images
        for index, _ in enumerate(self.ref_list):
            self.ref_file_list.extend([self.ref_list[index]])

        # Determine the length of the dataset
        if len(self.file_list)==len(self.ref_file_list):
            self.item_num = len(self.file_list)
        else:
            print('There are problems with the dataset. (Different number of reference and low-resolution images)')

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        ref_file_name = [self.ref_dataset_dir + self.ref_file_list[index]]

        with h5py.File(file_name[0], 'r') as hf1, h5py.File(ref_file_name[0], 'r') as hf2:

            Lr_SAI_y = np.array(hf1.get('Lr_SAI_y')).astype(np.float32)
            Hr_SAI_y = np.array(hf1.get('Hr_SAI_y')).astype(np.float32)
            Sr_SAI_cbcr = np.array(hf1.get('Sr_SAI_cbcr'), dtype='single')
            ref_SAI_y = np.array(hf2.get('ref_y')).astype(np.float32)  # ref_sai_y
            # print(ref_SAI_y.shap e)
            # ref_y = ref_SAI_y[:, :, self.random_number]
            ref_y = ref_SAI_y

            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))
            ref_y = np.transpose(ref_y, (1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())
        ref_y = ToTensor()(ref_y.copy())

        Lr_angRes_in = self.angRes
        Lr_angRes_out = self.angRes
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, ref_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num

def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    dataset_dir = args.path_for_test + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                      str(args.scale_factor) + 'x/'
    data_list = os.listdir(dataset_dir)

    test_Loaders = []

    test_Dataset = TestSetDataLoader(args)
    length_of_tests = len(test_Dataset)

    test_Loaders = DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                           str(args.scale_factor) + 'x/'
        self.ref_dataset_dir = args.path_for_test + 'ref_' + str(args.scale_factor) + 'x/'

        self.dataset_list = os.listdir(self.dataset_dir)
        self.ref_list = os.listdir(self.ref_dataset_dir)

        self.file_list = []
        self.ref_file_list = []

        # self.random_number = random.randint(0, 55)
        # The 'random_number'st in the 2D HR image stack is taken out as the 2D reference image
        # print(f"The '{self.random_number}'st in the 2D HR image stack is taken out as the 2D reference image")

        # input LR LF images
        for index, _ in enumerate(self.dataset_list):
            self.file_list.extend([self.dataset_list[index]])

        #  stack of 2D HR ref images
        for index, _ in enumerate(self.ref_list):
            self.ref_file_list.extend([self.ref_list[index]])

        # Determine the length of the dataset
        if len(self.file_list)==len(self.ref_file_list):
            self.item_num = len(self.file_list)
        else:
            print('There are problems with the dataset. (Different number of reference and low-resolution images)')

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        ref_file_name = [self.ref_dataset_dir + self.ref_file_list[index]]

        with h5py.File(file_name[0], 'r') as hf1, h5py.File(ref_file_name[0], 'r') as hf2:

            Lr_SAI_y = np.array(hf1.get('Lr_SAI_y')).astype(np.float32)
            Hr_SAI_y = np.array(hf1.get('Hr_SAI_y')).astype(np.float32)
            Sr_SAI_cbcr = np.array(hf1.get('Sr_SAI_cbcr'), dtype='single')
            ref_SAI_y = np.array(hf2.get('ref_y')).astype(np.float32)  # ref_sai_y
            # print(ref_SAI_y.shap e)
            # ref_y = ref_SAI_y[:, :, self.random_number]
            ref_y = ref_SAI_y

            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))
            ref_y = np.transpose(ref_y, (1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())
        ref_y = ToTensor()(ref_y.copy())

        Lr_angRes_in = self.angRes
        Lr_angRes_out = self.angRes
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y,ref_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation_8(data, label, ref, hr_down4, hr_down2):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
        ref = ref[:, ::-1]
        hr_down4 = hr_down4[:, ::-1]
        hr_down2 = hr_down2[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
        ref = ref[::-1, :]
        hr_down4 = hr_down4[::-1, :]
        hr_down2 = hr_down2[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
        ref = ref.transpose(1, 0)
        hr_down4 = hr_down4.transpose(1, 0)
        hr_down2 = hr_down2.transpose(1, 0)
    return data, label, ref, hr_down4, hr_down2

def augmentation_4(data, label, ref, hr_down2):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
        ref = ref[:, ::-1]
        hr_down2 = hr_down2[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
        ref = ref[::-1, :]
        hr_down2 = hr_down2[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
        ref = ref.transpose(1, 0)
        hr_down2 = hr_down2.transpose(1, 0)
    return data, label, ref, hr_down2

def augmentation(data, label, ref):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
        ref = ref[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
        ref = ref[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
        ref = ref.transpose(1, 0)
    return data, label, ref

