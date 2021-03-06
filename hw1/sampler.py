import numpy as np
import math
import matplotlib.pyplot as plt


class ProbabilityModel:
    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass

    def set_times(self, n):
        return np.array([self.sample() for _ in range(n)])

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
        self.ap = ap

    def sample(self):
        x = np.random.uniform(0, 1)
        thresh = self.ap[0]
        i = 0
        while x > thresh:
            i += 1
            thresh += self.ap[i]
        return self.ap[i]


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)


class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self, ap, pm):
        self.ap = ap
        self.Mu = pm[0]
        self.Sigma = pm[1]

    def sample(self):
        x = np.random.uniform(0, 1)
        thresh = self.ap[0]
        i = 0
        while x > thresh:
            i += 1
            thresh += self.ap[i]
        model = MultiVariateNormal(self.Mu[i], self.Sigma[i])
        return model.sample()


def plot_1():
    ap = [0.1, 0.1, 0.3, 0.3, 0.2]
    categorical_model = Categorical(ap)
    categorical_samples = categorical_model.set_times(1000)
    categories, counts = np.unique(categorical_samples, return_counts=True)
    plt.bar([str(i) for i in categories], counts)
    plt.title('Categorical distribution histogram')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.savefig('Categorical_distribution.png')
    plt.show()


def plot_2():
    mu = 1
    sigma = 1
    univariate = UnivariateNormal(mu, sigma)
    univariate_samples = univariate.set_times(1000)
    plt.hist(univariate_samples)
    plt.title('Univariate Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.savefig('Univariate_distribution.png')
    plt.show()


# plot_1()
# plot_2()

def plot_3():
    Mu = np.array([1, 1])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    multi_normal_model = MultiVariateNormal(Mu, Sigma)
    multi_normal_samples = multi_normal_model.set_times(1000)
    x = multi_normal_samples[:, 0]
    y = multi_normal_samples[:, 1]
    plt.scatter(x, y)
    plt.title('2-D Gaussian')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.savefig('GaussianScatterPlot.png')
    plt.show()

def plot_4():
    ap = np.array([0.25, 0.25, 0.25, 0.25])
    Mu = np.array([1, 1, 1, -1, -1, 1, -1, -1]).reshape(4, 2)
    Sigma = [np.identity(2) for _ in range(4)]
    mix_model = MixtureModel(ap, [Mu, Sigma])
    mix_samples = mix_model.set_times(5000)
    x = mix_samples[:, 0]
    y = mix_samples[:, 1]
    plt.title('Mixture of Four Gaussians in 2 dimensions')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.scatter(x, y)
    plt.savefig('MixtureGaussians.png')
    plt.show()
    size = 5000
    count = 0
    for _ in range(size):
        x, y = mix_model.sample()
        if (x - 0.1) ** 2 + (y - 0.2) ** 2 <= 1:
            count += 1
    return print(float(count / size))
# plot_3()
plot_4()


