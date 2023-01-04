import numpy as np
import matplotlib.pyplot as plt

from Bandit import Bandit
from Agent import Agent

ARMS = 50
ITERS = 1000

bandit = Bandit(ARMS)

for i in [0.1, 0.05, 0.01]:
    agent = Agent(i)

    averageReward = agent.play(bandit, ITERS)

    x = np.arange(ITERS + 1)
    plt.plot(x, averageReward, label=f"$\epsilon={i}$")

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Average reward")
plt.title("Average $\epsilon-greedy$ rewards after " + str(ITERS) 
    + " Episodes and " + str(ARMS) + " arms")
plt.show()
