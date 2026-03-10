import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm


def expected_improvement(mu, sigma, best):

    sigma = np.maximum(sigma, 1e-9)

    z = (mu - best) / sigma

    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


def bayesian_optimize(objective, bounds, n_init=5, n_iter=20):

    samples = []

    # random initialization
    for _ in range(n_init):

        params = [np.random.randint(low, high) for low, high in bounds]

        score = objective(*params)

        samples.append((params, score))


    for _ in range(n_iter):

        X = np.array([s[0] for s in samples])
        y = np.array([s[1] for s in samples])

        gp = GaussianProcessRegressor(kernel=RBF())

        gp.fit(X, y)

        best = y.max()

        best_x = None
        best_ei = -np.inf

        # random candidate search
        for _ in range(100):

            x = np.array([[np.random.randint(low, high) for low, high in bounds]])

            mu, sigma = gp.predict(x, return_std=True)

            ei = expected_improvement(mu, sigma, best)

            if ei > best_ei:

                best_ei = ei

                best_x = x[0]

        score = objective(*best_x)

        samples.append((best_x, score))


    best_sample = max(samples, key=lambda x: x[1])

    return best_sample