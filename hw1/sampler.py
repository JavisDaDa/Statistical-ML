import numpy as np
import math
class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    # Box-Muller Transform
    def sample(self):
        u_1 = np.random.uniform(0, 1)
        u_2 = np.random.uniform(0, 1)
        z_0 = math.sqrt(-2 * np.log(u_1)) * np.cos(np.pi * 2 * u_2)
        return self.sigma * z_0 + self.mu
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self, Mu, Sigma):
        self.Mu = Mu
        self.Sigma = Sigma

    def sample(self):
        uni_normal = UnivariateNormal(0, 1)
        # cholesky decomposition
        L = np.linalg.cholesky(self.Sigma)
        Z = np.array([uni_normal.sample() for _ in range(len(self.Mu))])
        return self.Mu + np.dot(L, Z)
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self, ap):
        pass

    def sample(self):
        pass


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self, ap, pm):
        pass

    def sample(self):
        pass