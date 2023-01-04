import numpy as np

from Bandit import Bandit

class Agent:
    def __init__(self, eps: int) -> None:
        self.eps = eps
        self.estimates = None
        self.count = None
        
    def select_an_arm(self, arms: int):
        r = np.random.rand()
        
        if r < self.eps:
            return np.random.randint(arms)
        
        return np.argmax(self.estimates)
    
    def play(self, bandit: Bandit, iters: int) -> float:
        self.estimates = np.zeros(len(bandit.performances))
        self.count = np.zeros(len(bandit.performances))
        average_rewards = np.zeros(iters + 1)
        
        for i in range(iters):
            a = self.select_an_arm(len(bandit.performances))
            reward = bandit.pull(a)
            
            self.count[a] += 1
            
            self.estimates[a] = self.estimates[a] + (1 / self.count[a]) * (reward - self.estimates[a])
            
            average_rewards[i+1] = average_rewards[i] + (1 / np.sum(self.count)) * (reward - average_rewards[i])
        
        return average_rewards
