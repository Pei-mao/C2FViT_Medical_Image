import numpy as np
import itertools

import nibabel as nib
import numpy as np
import torch
import torch.utils.data as Data
import csv
import torch.nn.functional as F


def update_affine(original_affine, crop_slices):
    """Updates the affine matrix after cropping."""
    translation = [slice.start for slice in crop_slices]
    new_affine = original_affine.copy()
    new_affine[:3, 3] = original_affine[:3, 3] + np.dot(original_affine[:3, :3], translation)
    return new_affine


def crop_image(image, target_shape):
    """Crops the image to the target shape."""
    current_shape = image.shape
    crop_slices = []

    for i in range(len(target_shape)):
        start = (current_shape[i] - target_shape[i]) // 2
        end = start + target_shape[i]
        crop_slices.append(slice(start, end))

    cropped_image = image[tuple(crop_slices)]
    return cropped_image, crop_slices


def reorient_image(image, target_orientation):    
    # Get the current affine
    affine = image.affine
    
    # Get the current orientation
    current_orientation = nib.orientations.aff2axcodes(affine)
    
    # Determine the transformation to LIA
    ornt_transform = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(current_orientation),
        nib.orientations.axcodes2ornt(target_orientation)
    )
    
    # Apply the orientation transformation
    reoriented_data = nib.orientations.apply_orientation(image.get_fdata(), ornt_transform)
    
    # Calculate the new affine matrix
    new_affine = nib.orientations.inv_ornt_aff(ornt_transform, reoriented_data.shape)
    new_affine = affine @ new_affine
    
    return nib.Nifti1Image(reoriented_data, new_affine, image.header)


def pad_to_shape(img, target_shape):
    """
    Pads the input image with zeros to match the target shape.
    """
    padding = [(max(0, t - s)) for s, t in zip(img.shape, target_shape)]
    pad_width = [(p // 2, p - (p // 2)) for p in padding]
    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img


def load_4D(name, RAS=False):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    # X = np.reshape(X, (1,)+ X.shape)
    X = nib.load(name)
    if RAS:
        X = reorient_image(X, ('R', 'A', 'S'))
    else:
        X = reorient_image(X, ('L', 'I', 'A'))
    X = X.get_fdata()
    
    # Ensure the image size is 256x256x256, pad if necessary
    target_shape = (256, 256, 256)
    if X.shape != target_shape:
        X = pad_to_shape(X, target_shape)
        X, _ = crop_image(X, target_shape=(256, 256, 256))
    
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_4D_channel(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.transpose(X, (3, 0, 1, 2))
    return X


def min_max_norm(img):
    max = np.max(img)
    min = np.min(img)

    norm_img = (img - min) / (max - min)

    return norm_img


def save_img(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_flow(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, names, labels, norm=True, use_label=False, RAS=True):
        'Initialization'
        self.names = names
        self.labels = labels
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))
        self.index_pair_label = list(itertools.permutations(labels, 2))
        self.use_label = use_label
        self.RAS = RAS

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0], self.RAS)
        img_B = load_4D(self.index_pair[step][1], self.RAS)

        img_A_label = load_4D(self.index_pair_label[step][0], self.RAS)
        img_B_label = load_4D(self.index_pair_label[step][1], self.RAS)

        if self.norm:
            img_A = min_max_norm(img_A)
            img_B = min_max_norm(img_B)

        if self.use_label:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(img_A_label).float(), torch.from_numpy(img_B_label).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_MNI152(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_list, label_list, fixed_img, fixed_label, need_label=True, RAS=True):
        'Initialization'
        super(Dataset_epoch_MNI152, self).__init__()
        # self.exp_path = exp_path
        self.img_pair = img_list
        self.label_pair = label_list
        self.need_label = need_label
        self.fixed_img = fixed_img
        self.fixed_label = fixed_label
        self.RAS = RAS

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.img_pair[step], self.RAS)
        fixed_img = load_4D(self.fixed_img, self.RAS)
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

        if self.need_label:
            moving_label = load_4D(self.label_pair[step], self.RAS)
            fixed_label = load_4D(self.fixed_label, self.RAS)
            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float(), torch.from_numpy(moving_label.copy()).float(), torch.from_numpy(fixed_label.copy()).float()
        else:
            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float()


class Dataset_epoch_MNI152_pre_one_hot(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_list, label_list, fixed_img, fixed_label, need_label=True):
        'Initialization'
        super(Dataset_epoch_MNI152_pre_one_hot, self).__init__()
        # self.exp_path = exp_path
        self.img_pair = img_list
        self.label_pair = label_list
        self.need_label = need_label
        self.fixed_img = fixed_img
        self.fixed_label = fixed_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.img_pair[step])
        fixed_img = load_4D(self.fixed_img)
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

        if self.need_label:
            moving_label = load_4D_channel(self.label_pair[step])
            fixed_label = load_4D_channel(self.fixed_label)

            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float(), torch.from_numpy(moving_label).float(), torch.from_numpy(fixed_label).float()
        else:
            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float()


class Dataset_epoch_onehot(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, names, labels, norm=True, use_label=False):
        'Initialization'
        self.names = names
        self.labels = labels
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))
        self.index_pair_label = list(itertools.permutations(labels, 2))
        self.use_label = use_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        img_A_label = load_4D_channel(self.index_pair_label[step][0])
        img_B_label = load_4D_channel(self.index_pair_label[step][1])

        if self.norm:
            img_A = min_max_norm(img_A)
            img_B = min_max_norm(img_B)

        if self.use_label:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(img_A_label).float(), torch.from_numpy(img_B_label).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()
