import numpy as np

class Bandit:
    def __init__(self, arms: int) -> None:
        self.performances = np.linspace(0, arms-1, arms)
    
    def pull(self, arm: int) -> float:
        return np.random.normal(self.performances[arm])
