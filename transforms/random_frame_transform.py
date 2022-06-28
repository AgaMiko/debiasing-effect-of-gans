import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import glob
import numpy


class RandomFrameTransform:
    """Places a frame image over the input. 
    Randomly rotates masks if rand_rotate is set to True"""

    def __init__(self, mask_list, im_size=256, rand_rotate=False,
                 p=1.0):
        self.mask_list = mask_list
        self.len = len(self.mask_list)
        self.p = float(p)
        self.im_size = im_size
        self.rand_rotate = rand_rotate

    def __call__(self, image,  **params):
        p = random.randint(0, 100)
        if p < self.p*100:
            mask_nr = random.randint(0, self.len-1)
            mask = Image.open(self.mask_list[mask_nr]).convert('L')
            mask = mask.resize((self.im_size, self.im_size), Image.ANTIALIAS)

            temp_image = Image.fromarray(image)
            temp_image = temp_image.resize(
                (self.im_size, self.im_size), Image.ANTIALIAS)

            if type(image).__module__ == 'numpy':
                image = Image.fromarray(image)
            temp_mask = image.copy()
            temp_mask.paste(mask)

            if self.rand_rotate:
                angle = random.randint(0, 360)
                temp_image = temp_image.rotate(angle=angle)
                mask = mask.rotate(angle=angle)

            temp_mask = temp_mask.resize(
                (self.im_size, self.im_size), Image.ANTIALIAS)
            temp_image = Image.composite(
                image1=temp_image, image2=temp_mask, mask=mask)
            return numpy.array(temp_image, dtype=numpy.dtype("uint8"))
        else:
            return image
