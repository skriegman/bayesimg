import numpy as np
from scipy import stats
from copy import deepcopy
from PIL import Image
from scipy import ndimage, misc
from itertools import product
#import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import BlendedGenericTransform


#sns.set(color_codes=True, context="poster")
#sns.set_style("white")
#sns.set_palette("Set2", 10)


ORIGINAL_IMAGE = "img/true_img_bw.png"  
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

    subtraction = np.asarray(clone - noise, dtype=np.uint8)
    addition = np.asarray(clone + noise, dtype=np.uint8)

    boolz = np.array(clone == 255)
    result = np.where(boolz, subtraction, addition)

    plt.imshow(result, cmap=plt.get_cmap('gray'))
    plt.show()
    return result


def empirical(bw_img, neighborhood):
    num_neighbors = ndimage.generic_filter(bw_img, np.count_nonzero, footprint=neighborhood)
    distribution = {}
    for n in range(neighborhood.sum() + 1):
        distribution[n] = np.mean(bw_img[np.where(num_neighbors == n)]) / 255.0
    return distribution


def sum_neighbors(coords, padded_img):
    x, y = coords
    neighborhood = np.array([(x, y+1), (x, y-1), (x+1, y), (x-1, y)])
    return np.sum(padded_img[neighborhood])


def metropolis(y, sigma, max_iter, beta, plot=True):

    y = y[50:100, 100:150]
    y_pad = np.pad(y, 1, 'constant')
    coords = [(row, col) for row, col in product(range(y.shape[0]), range(y.shape[1]))]
    y = y.flatten()  # todo: make sure this is in the same order as coords

    if plot:
        plt.ion()
    # y is noisy image
    x = np.round(np.random.random(y.shape), 0)

    accept_count = 0
    evaluate_count = 0

    for t in range(max_iter):
        sites_to_visit = [a for a in range(y.size)]

        if t % 50 == 0:
            print "Iteration " + str(t)

        for pixel in range(y.size):
            evaluate_count += 1

            if plot:
                plt.imshow(x)
                plt.pause(0.05)

            i = sites_to_visit.pop(np.random.randint(0, len(sites_to_visit)))
            # coords = np.unravel_index(i, dims=y.shape)  # this is a tuple

            y_i = y[i]
            x_i = x[i]
            x_i_prime = 1 - x_i
            k_b = sum_neighbors(coords[i], y_pad)
            k_w = 4 - k_b

            k = k_b if x_i else k_w
            d = beta * k - (y_i - x_i)**2 / float(2 * sigma**2)

            k = k_b if x_i_prime else k_w
            d_prime = beta * k - (y_i - x_i_prime) ** 2 / float(2 * sigma ** 2)

            u = np.random.random()
            p = np.exp(min(d_prime - d, 0))

            if u < p:
                x[i] = x_i_prime
                accept_count += 1
                # print(coords)

    return x, accept_count / evaluate_count

sigma = 50
orig_img_bw = misc.imread(ORIGINAL_IMAGE)
noisy_img = copy_with_gaussian_noise(orig_img_bw, sigma=sigma)
misc.imsave("img/noisy_bw_{}.png".format(sigma), noisy_img)

y = misc.imread("img/noisy_bw_{}.png".format(sigma))


plt.imshow(orig_img_bw[50:100, 100:150], cmap=plt.get_cmap('gray'))
plt.show()

plt.imshow(noisy_img[50:100, 100:150], cmap=plt.get_cmap('gray'))
plt.show()


x, accept_rate = metropolis(y, sigma, max_iter=1000, beta=0.7, plot=False)
print "accept rate: " + str(accept_rate)
y = y[50:100, 100:150]
print x.size
print y.shape
x = np.reshape(x, y.shape)
plt.imshow(x, cmap=plt.get_cmap('gray'))
plt.show()

print "omg we finished"
