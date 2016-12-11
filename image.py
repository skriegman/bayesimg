import numpy as np
import time
from copy import deepcopy
from PIL import Image
from scipy import ndimage, misc
import matplotlib.pyplot as plt


def convert_to_black_white(filename="img/true_img.png", new_name="img/true_img_bw.png"):
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
    bottom_edge = idx >= x_lim*y_lim - x_lim

    neighborhood_dict = {"left": idx-1,
                         "right": idx+1,
                         "top": idx-x_lim,
                         "bottom": idx+x_lim}

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
    return T_0 * eda**(t - 1)


def metropolis(y, sigma, beta, max_iter=1000000, max_time=1000000, save_every=100, MAP=False, burn_in=1):

    # y is noisy image, x is the binary guess

    y = y / 255.  # normalize
    sigma /= 255.

    start_time = time.time()

    x_lim, y_lim = y.shape
    y = y.flatten()
    x = np.round(np.random.random(y.shape), 0)
    saved_x = np.reshape(x, (1, y.size))

    accept_count = 0
    evaluate_count = 0 

    for t in range(max_iter):

        # check the time
        if time.time() - start_time > max_time:
            return saved_x, float(accept_count) / float(evaluate_count)

        sites_to_visit = [a for a in range(y.size)]

        if t % 100 == 0:
            print("Iteration " + str(t))

        for pixel in range(y.size):
            evaluate_count += 1

            # save image
            if pixel % save_every == 0:
                name = "MAP" if MAP else "MPM"
                misc.imsave("{0}/{0}_{1}_{2}.png".format(name, t, pixel), x.reshape(x_lim, y_lim))

            i = sites_to_visit.pop(np.random.randint(0, len(sites_to_visit)))

            y_i = y[i]
            x_i = x[i]
            x_i_prime = 1 - x_i

            k_b, num_neighbors = sum_neighbors(i, x, x_lim, y_lim)
            k_w = num_neighbors - k_b

            k = k_b if x_i else k_w
            d = beta * k - (y_i - x_i)**2 / float(2 * sigma**2)

            k = k_b if x_i_prime else k_w
            d_prime = beta * k - (y_i - x_i_prime) ** 2 / float(2 * sigma ** 2)

            u = np.random.random()
            
            if MAP:
                p = np.exp(min(d_prime - d, 0) / temperature(t + 1))  # f simulated annealing (MAP estimator)
            else:
                p = np.exp(min(d_prime - d, 0))

            if u < p:
                x[i] = x_i_prime
                accept_count += 1

        if t >= burn_in:
            saved_x = np.concatenate((saved_x, np.reshape(x, (1, y.size))))

    return saved_x, float(accept_count) / float(evaluate_count)


def traceplot(pixel_sample):
    x = np.array(range(0, pixel_sample.size))
    plt.plot(x, pixel_sample)
    plt.title("Tracplot")


def MPM(samples, shape):
    '''Marginal Posterior Modes (MPM) estimator, which
    is defined as
     0, if Prob (xi = 1|y) > 1/2
     1, if Prob (xi = 1|y) =< 1/2
    and is easily calculated by counting the number of times xi is equal to 1.
    '''
    sums = np.sum(samples, axis=0)
    boolz = np.array(sums > samples.shape[0]/2)
    print(samples.shape[1])
    image = np.reshape(np.where(boolz, 0, 1), shape)
    plt.imshow(image)
    misc.imsave("img/MPM_estimate.png", 1-image)
    plt.show()
    return image


MAP = True
burn_in = 0  # set to zero for MAP movie
sigma = 90
beta = 0.7  # 0.6 for MAP (?)
save_every = 10000
max_iter = 5000


# # convert_to_black_white()
# orig_img_bw = misc.imread("img/true_img_bw.png")
# noisy_img = copy_with_gaussian_noise(orig_img_bw, sigma=sigma)
# misc.imsave("img/noisy_bw_{}.png".format(sigma), noisy_img)

# result, accept_rate = metropolis(noisy_img, sigma=sigma, beta=beta, max_iter=max_iter, MAP=MAP, save_every=save_every,
#                                  burn_in=burn_in)
# print("accept rate: " + str(accept_rate))
#
# MPMimage = MPM(result, noisy_img.shape)


orig_img = misc.imread("img/true_img_bw.png")
MAP = misc.imread("img/MAP_estimate_90.png")
MPM = misc.imread("img/MPM_estimate_90.png")
print MAP

map_error = np.mean(np.abs(orig_img - MAP))
mpm_error = np.mean(np.abs(orig_img - MPM))

print map_error/255., mpm_error/255.
