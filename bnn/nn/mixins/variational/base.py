from abc import abstractmethod


from ..base import BayesianMixin


class VariationalMixin(BayesianMixin):

    def parameter_loss(self):
        return self.kl_divergence()

    @abstractmethod
    def kl_divergence(self):
        raise NotImplementedError

