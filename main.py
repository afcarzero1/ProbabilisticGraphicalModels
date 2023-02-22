from modeling.MCHastings import MCHastings, NormalProposalDistribution
from modeling.MixureGaussian import MixtureIndependentGaussian

import matplotlib.pyplot as plt
import numpy as np


def MixtureGaussianMCHastings():

    # Instantiate the true distribution and transition distribution
    distribution = MixtureIndependentGaussian(amplitudes=[0.5, 0.5],
                                              mus=[0, 3],
                                              sigmas=[1, 0.5])

    sampler = MCHastings(true_distribution=distribution,
                         transition_distribution=NormalProposalDistribution(sigma=1))

    # Sample from them
    samples = sampler.execute(initial_value=0, number_samples=10000)

    # plot them
    plt.hist(samples, bins=50, density=True, alpha=0.5)

    x = np.linspace(-5, 5, 100)
    plt.plot(x, [distribution.p(n) for n in x], 'r', lw=2)
    plt.ylabel("Density")
    plt.xlabel("x")
    plt.title("Samples distribution")
    plt.show()




def main():
    MixtureGaussianMCHastings()


if __name__ == '__main__':
    main()
