import numpy as np
import time
from scipy import misc

from utils import sum_neighbors, temperature


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
            d = beta * k - (y_i - x_i) ** 2 / float(2 * sigma ** 2)

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
