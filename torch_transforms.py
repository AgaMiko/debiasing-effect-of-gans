from torchvision import transforms
import numpy as np
from transforms.random_hair_transform import RandomHairTransform
from transforms.random_frame_transform import RandomFrameTransform


def get_augmentation(transform):
    return lambda img:np.array(img)

def get_transforms(image_size, type_aug='frame', aug_p=1.0, mask_list=""):
    if type_aug in ["short", "medium", "dense", "ruler"]:
        transforms_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            RandomHairTransform(p=aug_p, mask_list=mask_list),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225]),
        ])
    elif type_aug == "frame":
        transforms_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        RandomFrameTransform(p=aug_p, mask_list=mask_list),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]),
    ])
    transforms_val = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]),
    ])
    return transforms_train, transforms_val
