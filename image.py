import numpy as np
from PIL import Image
from scipy import ndimage, misc


ORIGINAL_IMAGE = "img/character_bw.png"  # "img/character.jpg"
NEIGHBORHOOD = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]])


def convert_to_black_white(filename):
    color_img = Image.open(filename)
    gray_img = color_img.convert('L')
    black_white_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')
    black_white_img.save("img/character_bw.png")


def add_bitflip_noise(img, prob_flip):
    clone = img.copy()
    flip_index = np.random.random(clone.shape) < prob_flip
    clone[flip_index] = [255 if x == 0 else 0 for x in clone[flip_index]]
    return clone


def empirical(bw_img, neighborhood):
    num_neighbors = ndimage.generic_filter(bw_img, np.count_nonzero, footprint=neighborhood)
    distribution = {}
    for n in range(neighborhood.sum() + 1):
        distribution[n] = np.mean(bw_img[np.where(num_neighbors == n)]) / 255.0
    return distribution


orig_img_bw = misc.imread(ORIGINAL_IMAGE)
noisey_img = add_bitflip_noise(orig_img_bw, 0.4)
misc.imsave("img/noisey_bw.png", noisey_img)

print empirical(orig_img_bw, NEIGHBORHOOD)

