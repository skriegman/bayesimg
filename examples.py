import numpy as np
from scipy import misc

from utils import convert_to_black_white, copy_with_gaussian_noise, MPM
from metropolis import metropolis

SEED = 1
MAP = True
burn_in = 0
sigma = 90
beta = 0.7
save_every = 1000000000
max_iter = 1000

np.random.seed(SEED)

# convert_to_black_white()

orig_img_bw = misc.imread("example_data/true_img_bw.png")
noisy_img = copy_with_gaussian_noise(orig_img_bw, sigma=sigma)
misc.imsave("example_data/noisy_bw_{}.png".format(sigma), noisy_img)

result, accept_rate = metropolis(noisy_img, sigma=sigma, beta=beta, max_iter=max_iter, MAP=MAP, save_every=save_every,
                                 burn_in=burn_in)
print("accept rate: " + str(accept_rate))

MPMimage = MPM(result, noisy_img.shape)


# orig_img = misc.imread("example_data/true_img_bw.png")
# MAP = misc.imread("example_data/MAP_estimate_90.png")
# MPM = misc.imread("example_data/MPM_estimate_90.png")
# print MAP
#
# map_error = np.mean(np.abs(orig_img - MAP))
# mpm_error = np.mean(np.abs(orig_img - MPM))
#
# print map_error / 255., mpm_error / 255.
