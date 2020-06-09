from PIL import Image, ImageEnhance, ImageOps
import numpy as np

FILLCOLOR = (0, 0, 0)

def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude, fillcolor=FILLCOLOR)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


func_numpy = {
    "rotate": lambda img, magnitude: ndimage.rotate(img, magnitude),
    "translateX": lambda img, magnitude: ndimage.shift(img, [0, magnitude * img.shape[1] * random.choice([-1, 1]), 0]),
    "translateY": lambda img, magnitude: ndimage.shift(img, [magnitude * img.shape[1] * random.choice([-1, 1]),0, 0]),
}

func_pil = {
    "shearX": lambda img, magnitude: img.transform(
        img.size, Image.AFFINE, (1, magnitude * np.random.choice([-1, 1]), 0, 0, 1, 0),
        Image.BICUBIC, fillcolor=FILLCOLOR),
    "shearY": lambda img, magnitude: img.transform(
        img.size, Image.AFFINE, (1, 0, 0, magnitude * np.random.choice([-1, 1]), 1, 0),
        Image.BICUBIC, fillcolor=FILLCOLOR),
    "translateX": lambda img, magnitude: img.transform(
        img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * np.random.choice([-1, 1]), 0, 1, 0),
        fillcolor=FILLCOLOR),
    "translateY": lambda img, magnitude: img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * np.random.choice([-1, 1])),
        fillcolor=FILLCOLOR),
    "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
    "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * np.random.choice([-1, 1])),
    "posterize": lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
    "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
    "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
        1 + magnitude * np.random.choice([-1, 1])),
    "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
        1 + magnitude * np.random.choice([-1, 1])),
    "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
        1 + magnitude * np.random.choice([-1, 1])),
    "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
    "equalize": lambda img, magnitude: ImageOps.equalize(img),
}

strong_augment_bounds = {
    "autocontrast": [0, 1],
    "equalize": [0, 1],
    "rotate": [-30, 30],
    "solarize": [0, 256],
    "color": [0.05, 0.95],
    "contrast": [0.05, 0.95],
    "brightness": [0.05, 0.95],
    "sharpness": [0.05, 0.95],
    "shearX": [-0.1, 0.1],
    "translateX": [-0.1, 0.1],
    "translateY": [-0.1, 0.1],
    "posterize": [4, 8],
    "shearY": [-0.1, 0.1],
}

def strong_augmentation(im, is_array=False):
    if is_array:
        augmented_im = Image.fromarray(im)
    else:
        augmented_im = im
    augmentations = np.random.choice(list(func_pil.keys()), 4, replace=False)
    for augmentation_name in augmentations:
        p = np.random.uniform(strong_augment_bounds[augmentation_name][0], strong_augment_bounds[augmentation_name][1])
        augmented_im = func_pil[augmentation_name](augmented_im, p)
    if is_array:
        augmented_im = np.array(augmented_im)
    return augmented_im