import numpy as np
from copy import deepcopy
from PIL import Image
from scipy import ndimage, misc
import matplotlib.pyplot as plt


def convert_to_black_white(filename="example_data/true_img.png", new_name="example_data/true_img_bw.png"):
    color_img = Image.open(filename)
    gray_img = color_img.convert('L')
    black_white_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')
    black_white_img.save(new_name)


def copy_with_flip_noise(bw_img, prob_flip):
    clone = deepcopy(bw_img)
    flip_index = np.random.random(clone.shape) < prob_flip
    clone[flip_index] = [255 if x == 0 else 0 for x in clone[flip_index]]
    return clone


def copy_with_gaussian_noise(img, sigma=50):
    clone = deepcopy(img)
    noise = np.abs(np.random.normal(0, scale=sigma, size=img.shape))

    subtraction = np.asarray(clone - noise, dtype=np.uint8)
    addition = np.asarray(clone + noise, dtype=np.uint8)

    boolz = np.array(clone == 255)
    result = np.where(boolz, subtraction, addition)

    # plt.imshow(result, cmap=plt.get_cmap('gray'))
    # plt.show()
    return result


def empirical(bw_img, neighborhood):
    num_neighbors = ndimage.generic_filter(bw_img, np.count_nonzero, footprint=neighborhood)
    distribution = {}
    for n in range(neighborhood.sum() + 1):
        distribution[n] = np.mean(bw_img[np.where(num_neighbors == n)]) / 255.0
    return distribution


def sum_neighbors(idx, img, x_lim, y_lim):
    left_edge = idx % x_lim == 0
    right_edge = idx % x_lim == x_lim - 1
    top_edge = idx < x_lim
    bottom_edge = idx >= x_lim * y_lim - x_lim

    neighborhood_dict = {"left": idx - 1,
                         "right": idx + 1,
                         "top": idx - x_lim,
                         "bottom": idx + x_lim}

    if left_edge:
        neighborhood_dict.pop("left")
    if right_edge:
        neighborhood_dict.pop("right")
    if top_edge:
        neighborhood_dict.pop("top")
    if bottom_edge:
        neighborhood_dict.pop("bottom")

    neighbors_indices = np.array([v for k, v in neighborhood_dict.items()])

    return np.sum(img[neighbors_indices]), len(neighbors_indices)


def temperature(t):
    T_0 = 4
    eda = 0.999
    return T_0 * eda ** (t - 1)


def MPM(samples, shape):
    """Marginal Posterior Modes (MPM) estimator, which is defined as
        0, if Prob (xi = 1|y) > 1/2
        1, if Prob (xi = 1|y) =< 1/2
    and is easily calculated by counting the number of times xi is equal to 1.

    """
    sums = np.sum(samples, axis=0)
    boolz = np.array(sums > samples.shape[0] / 2)
    print(samples.shape[1])
    image = np.reshape(np.where(boolz, 0, 1), shape)
    plt.imshow(image)
    misc.imsave("example_data/MPM_estimate.png", 1 - image)
    plt.show()
    return image
