from abc import ABC
from typing import List, Any

from .MCHastings import Distribution
from scipy.stats import norm


class MixtureIndependentGaussian(Distribution):
    def __init__(self,
                 amplitudes: List[float],
                 mus: List[float],
                 sigmas: List[float]):
        """
        Initialize the mixure of gaussians
        :param amplitudes: The amplitudes of the Gaussians
        :type amplitudes: List[float]
        :param mus: The means of the gaussians
        :type mus: List[float]
        :param sigmas: The standard deviation of the Gaussians
        :type sigmas: List[float]
        """
        self.mus = mus
        self.sigmas = sigmas
        self.amplitudes = amplitudes


    def p(self,
          x: float
          ) -> float:
        p = 0
        for mu, sigma, amplitude in zip(self.mus,self.sigmas,self.amplitudes):
            p += norm.pdf(x, loc=mu, scale=sigma) * amplitude
        return p

    def sample(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
