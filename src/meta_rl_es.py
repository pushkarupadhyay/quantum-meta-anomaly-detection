import numpy as np
from sklearn.metrics import f1_score

class EvolutionStrategies:
    def __init__(self, params_shape, population_size=32, sigma=0.1, learning_rate=0.2):
        self.params_shape = params_shape
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=params_shape)

    def sample_population(self):
        epsilons = [np.random.randn(*self.params_shape) for _ in range(self.population_size)]
        population = [self.params + self.sigma * eps for eps in epsilons]
        return population, epsilons

    def update(self, rewards, epsilons):
        # Rewards should be a list or array of the same length as population_size
        rewards = np.array(rewards)
        # Standardize rewards
        if np.std(rewards) > 0:
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        
        gradient = np.zeros_like(self.params)
        for i in range(self.population_size):
            gradient += rewards[i] * epsilons[i]
        
        self.params += self.learning_rate / (self.population_size * self.sigma) * gradient
        return self.params

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
