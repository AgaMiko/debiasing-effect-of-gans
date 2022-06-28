import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import numpy


class RandomHairTransform:
    """Using hair masks, cut he hair from the source image, 
    and places it on the input image. Randomly rotates masks
    if rand_rotate is set to True"""

    def __init__(self, mask_list,
                 im_dir,
                 im_size=256,
                 rand_rotate=False,
                 p=0.5):
        self.mask_list = mask_list
        self.im_dir = im_dir
        self.len = len(self.mask_list)
        self.p = float(p)
        self.im_size = im_size
        self.rand_rotate = rand_rotate

    def __call__(self, image, rand_rotate=False, **params):
        p = random.randint(0, 100)
        if p < self.p*100:
            if type(image).__module__ == 'numpy':
                image = Image.fromarray(image)

            hair_nr = random.randint(0, self.len-1)
            im_path = self.im_dir + self.mask_list[hair_nr].split("/")[-1]
            mask = Image.open(self.mask_list[hair_nr]).convert('L')
            im = Image.open(im_path).convert('RGB')

            if self.rand_rotate:
                angle = random.randint(0, 360)
                im = im.rotate(angle=angle)
                mask = mask.rotate(angle=angle)

            source_image = image.resize(
                (self.im_size, self.im_size), Image.ANTIALIAS)
            mask = mask.resize((self.im_size, self.im_size), Image.ANTIALIAS)
            im = im.resize((self.im_size, self.im_size), Image.ANTIALIAS)

            temp_image = Image.composite(im, source_image, mask)
            return numpy.array(temp_image, dtype=numpy.dtype("uint8"))
        else:
            return image
