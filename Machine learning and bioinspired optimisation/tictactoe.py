import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.MultiBinary(9)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self._get_observation()

    def step(self, action):
        if self.board[action // 3][action % 3] != 0:
            return self._get_observation(), -1, True, {}
        self.board[action // 3][action % 3] = self.current_player
        done, winner = self._game_over()
        if done:
            if winner == 0:
                return self._get_observation(), 0, True, {}
            elif winner == 1:
                return self._get_observation(), 1, True, {}
            else:
                return self._get_observation(), -1, True, {}
        self.current_player = -self.current_player
        return self._get_observation(), 0, False, {}

    def render(self, mode='human'):
        print(self.board)

    def _get_observation(self):
        return np.array(self.board).flatten()

    def _game_over(self):
        for i in range(3):
            if self.board[i][0] != 0 and self.board[i][0] == self.board[i][1] and self.board[i][1] == self.board[i][2]:
                return True, self.board[i][0]
            if self.board[0][i] != 0 and self.board[0][i] == self.board[1][i] and self.board[1][i] == self.board[2][i]:
                return True, self.board[0][i]
        if self.board[0][0] != 0 and self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2]:
            return True, self.board[0][0]
        if self.board[0][2] != 0 and self.board[0][2] == self.board[1][1] and self.board[1][1] == self.board[2][0]:
            return True, self.board[0][2]
        if np.sum(np.abs(self.board)) == 9:
            return True, 0
        return False, 0
env = TicTacToeEnv()
observation = env.reset()
done = False
while not done:
    action = np.random.randint(9)
    observation, reward, done, info = env.step(action)
    env.render()
print("Game over!")
