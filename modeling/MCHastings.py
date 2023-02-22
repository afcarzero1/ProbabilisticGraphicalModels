from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
from scipy.stats import norm


class Distribution(ABC):
    """
    A class representing a distribution
    """

    @abstractmethod
    def p(self, *args, **kwrags) -> float:
        """
        Returns the probability of a given sample

        Args:
            *args ([Any]) : Any argument necessary
            **kwargs (Dict[Any]) : Any keyword argument necessary

        Returns:
            probability (float) : The probability of this sample according to this distribution
        """
        pass

    def sample(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


class NormalProposalDistribution(Distribution):

    def __init__(self, sigma: float):
        self._sigma = sigma

    def p(self, next_value: float, current_value: float) -> float:
        return norm.pdf(next_value, loc=current_value, scale=self._sigma)

    def sample(self, x_prev) -> Any:
        return np.random.normal(x_prev, self._sigma)


class MCHastings:
    def __init__(self,
                 true_distribution: Distribution,
                 transition_distribution: Distribution):
        """
        Initialize the MC hastings algorithm with the necessary distributions.

        Args:
            true_distribution (Distribution) : The true distribution
            transition_distribution (Distribution) : The distribution used to sample next
        """
        self.true_distribution = true_distribution
        self.transition_distribution = transition_distribution

    def execute(self,
                initial_value: Any,
                number_samples: int = 100
                ) -> List[Any]:
        """
        Perform an execution of the MC hastings algorithm for obtaining samples.

        Args:
            initial_value (Any): The initial value of the samples.
            number_samples (int): The number of samples to obtain.
        Returns:
             samples (List[Any]) : The samples.
        """

        current_value = initial_value
        samples = [current_value]

        accepted_samples = 0

        for _ in range(1, number_samples):

            # Sample from the proposal distribution
            proposal_next_value = self.transition_distribution.sample(current_value)

            # Compute quantities for comparisons
            numerator = self.true_distribution.p(proposal_next_value) * self.transition_distribution.p(current_value,
                                                                                                       proposal_next_value)
            denominator = self.true_distribution.p(current_value) * self.transition_distribution.p(proposal_next_value,
                                                                                                   current_value)
            # Compute the acceptance probability
            acceptance_prob = np.min((1., numerator / denominator))

            # Sample from a uniform
            u = np.random.uniform()
            if u < acceptance_prob:
                # if accepted update
                current_value = proposal_next_value
                accepted_samples+=1

            # Use the sample (either same as before or next one)
            samples.append(current_value)

        acceptance_rate = accepted_samples / number_samples

        return samples
