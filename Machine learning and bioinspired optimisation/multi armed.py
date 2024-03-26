import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k):
        self.k = k
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=k)
    
    def step(self, action):
        return np.random.normal(loc=self.q_star[action], scale=1.0)

class EpsilonGreedyAgent:
    def __init__(self, k, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
    
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])

def run_experiment(n_pulls, n_bandits, epsilon=0.1, alpha=0.1):
    rewards = np.zeros((n_bandits, n_pulls))
    for i in range(n_bandits):
        bandit = Bandit(k=8)
        agent = EpsilonGreedyAgent(k=8, epsilon=epsilon, alpha=alpha)
        for j in range(n_pulls):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            rewards[i, j] = reward
    return np.mean(rewards, axis=0)


n_pulls = 1000
n_bandits = 2000

rewards = run_experiment(n_pulls, n_bandits)

plt.plot(rewards)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Epsilon-Greedy Algorithm')
plt.show()
