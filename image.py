import numpy as np
from scipy import stats
from copy import deepcopy
from PIL import Image
from scipy import ndimage, misc
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import BlendedGenericTransform


sns.set(color_codes=True, context="poster")
sns.set_style("white")
sns.set_palette("Set2", 10)


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


def copy_with_gaussian_noise(img, sigma=50):
    clone = deepcopy(img)
    noise = np.abs(np.random.normal(0, scale=sigma, size=img.shape))
    clone += np.asarray(noise, dtype=np.uint8)

    # TODO: bounce back out of bounds noise (over 255)
    # out_bounds = clone > 255
    # clone[out_bounds] -= 2 * noise[out_bounds]

    return clone


def empirical(bw_img, neighborhood):
    num_neighbors = ndimage.generic_filter(bw_img, np.count_nonzero, footprint=neighborhood)
    distribution = {}
    for n in range(neighborhood.sum() + 1):
        distribution[n] = np.mean(bw_img[np.where(num_neighbors == n)]) / 255.0
    return distribution


def sum_neighbors(img, coords):
    padded_img = np.pad(img, 1, 'constant')
    x, y = coords
    neighborhood = np.array([(x, y+1), (x, y-1), (x+1, y), (x-1, y)])
    return np.sum(padded_img[neighborhood])


def metropolis(y, sigma, max_iter, beta, plot=True):

    y = y[50:100, 100:150]

    if plot:
        plt.ion()
    # y is noisy image
    x = np.round(np.random.random(y.shape), 0)

    for t in range(max_iter):
        sites_to_visit = [a for a in range(y.size)]

        for pixel in range(y.size):
            if plot:
                plt.imshow(x)
                plt.pause(0.05)

            i = sites_to_visit.pop(np.random.randint(0, len(sites_to_visit)))
            coords = np.unravel_index(i, dims=y.shape)

            y_i = y[coords]
            x_i = x[coords]
            x_i_prime = 1 - x_i
            k_b = sum_neighbors(y, coords)
            k_w = 4 - k_b

            k = k_b if x_i else k_w
            d = beta * k - (y_i - x_i)**2 / float(2 * sigma**2)

            k = k_b if x_i_prime else k_w
            d_prime = beta * k - (y_i - x_i_prime) ** 2 / float(2 * sigma ** 2)

            u = np.random.random()
            p = np.exp(min(d_prime - d, 0))

            if u < p:
                x[coords] = x_i_prime
                print coords


sigma = 50
orig_img_bw = misc.imread(ORIGINAL_IMAGE)
noisy_img = copy_with_gaussian_noise(orig_img_bw, sigma=sigma)
misc.imsave("img/noisy_bw_{}.png".format(sigma), noisy_img)

y = misc.imread("img/noisy_bw_{}.png".format(sigma))

metropolis(y, sigma, max_iter=1000, beta=50)

