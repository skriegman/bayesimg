import numpy as np
from copy import deepcopy
from PIL import Image
from scipy import ndimage, misc


ORIGINAL_IMAGE = "img/true_img_bw.png"  # "img/true_img.jpg"
NEIGHBORHOOD = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]])


def convert_to_black_white(filename="img/true_img.jpg", new_name="img/true_img_bw.png"):
    color_img = Image.open(filename)
    gray_img = color_img.convert('L')
    black_white_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')
    black_white_img.save(new_name)


def copy_with_flip_noise(bw_img, prob_flip):
    clone = deepcopy(bw_img)
    flip_index = np.random.random(clone.shape) < prob_flip
    clone[flip_index] = [255 if x == 0 else 0 for x in clone[flip_index]]
    return clone


def copy_with_gaussian_noise(img, prob_flip, loc=128, scale=50):
    clone = deepcopy(img)
    noise_index = np.random.random(clone.shape) < prob_flip
    clone[noise_index] += np.asarray(np.random.normal(loc, scale, size=np.sum(noise_index)), dtype=np.uint8)
    return clone


def empirical(bw_img, neighborhood):
    num_neighbors = ndimage.generic_filter(bw_img, np.count_nonzero, footprint=neighborhood)
    distribution = {}
    for n in range(neighborhood.sum() + 1):
        distribution[n] = np.mean(bw_img[np.where(num_neighbors == n)]) / 255.0
    return distribution


orig_img_bw = misc.imread(ORIGINAL_IMAGE)
# print empirical(orig_img_bw, NEIGHBORHOOD)

p = 0.85
noisey_img = copy_with_gaussian_noise(orig_img_bw, p)
misc.imsave("img/noisey_bw_{}.png".format(int(p*100)), noisey_img)

