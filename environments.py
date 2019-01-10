from typing import Tuple
import numpy as np
import math

class Scenario:
    def __init__(self):
        self.s = 0

    def resetState(self):
        pass

    def takeAction(self, a_idx: int) -> float:
        r = 0.0
        return r

class Gridworld(Scenario):
    def __init__(self):
        self.grid_labels = [
            [None , None , None , None , None , None , None]  ,
            [None , 0    , 1    , 2    , 3    , 4    , None]  ,
            [None , 5    , 6    , 7    , 8    , 9    , None]  ,
            [None , 10   , 11   , None , 12   , 13   , None]  ,
            [None , 14   , 15   , None , 16   , 17   , None]  ,
            [None , 18   , 19   , 20   , 21   , 22   , None]  ,
            [None , None , None , None , None , None , None]]

        self.grid_rewards = [
            [0 , 0 , 0 , 0   , 0 , 0  , 0]  ,
            [0 , 0 , 0 , 0   , 0 , 0  , 0]  ,
            [0 , 0 , 0 , 0   , 0 , 0  , 0]  ,
            [0 , 0 , 0 , 0   , 0 , 0  , 0]  ,
            [0 , 0 , 0 , 0   , 0 , 0  , 0]  ,
            [0 , 0 , 0 , -10 , 0 , 10 , 0]  ,
            [0 , 0 , 0 , 0   , 0 , 0  , 0]]

        self.state_neighbors = {}
        self.state_rewards = {}
        for i in range(1,5+1):
            for j in range(1, 5+1):
                if self.grid_labels[i][j] is not None:
                    key = self.grid_labels[i][j]
                    self.state_neighbors[key] = {
                        "L": self.grid_labels[i][j-1],
                        "U": self.grid_labels[i-1][j],
                        "R": self.grid_labels[i][j+1],
                        "D": self.grid_labels[i+1][j],
                    }
                    self.state_rewards[key] = self.grid_rewards[i][j]

        self.actions = ["L", "U", "R", "D"]
        self.trans_types = ["Success", "Fail", "VR", "VL"]
        self.trans_probs = [0.80, 0.10, 0.05, 0.05]
        self.gamma = 0.9
        self.start_state = 0
        self.goal_state = 22
        self.num_states = 23

        self.resetState()

    def resetState(self):
        self.s = self.start_state
        self.discount = 1.0
        self.r_cum = 0.0

    def takeAction(self, a_idx: int) -> float:
        # Determine where the robot would go based on taken action and Pr(success)
        trans_result = np.random.choice(a=self.trans_types, p=self.trans_probs)

        if trans_result != "Fail":
            if trans_result == "Success":
                a = self.actions[a_idx]
            elif trans_result == "VR":
                a = self.actions[(a_idx + 1) % len(self.trans_types)]
            elif trans_result == "VL":
                a = self.actions[(a_idx - 1) % len(self.trans_types)]

            # If the transition valid, do it
            if self.state_neighbors[self.s][a] is not None:
                self.s = self.state_neighbors[self.s][a]

        # Return discounted reward at resulting state
        r = self.discount * self.state_rewards[self.s] 
        self.r_cum += r
        self.discount *= self.gamma

        return r

class MountainCar(Scenario):
    def __init__(self):
        self.start_x = -0.5
        self.start_v = 0.0
        self.x_lb = -1.2
        self.x_ub = 0.5
        self.v_lb = -0.07
        self.v_ub = 0.07
        self.goal_state = np.array([0.5, 0.0])
        self.gamma = 1.0

        self.actions = [-1, 0, 1]
        self.num_state_vars = 2

        self.resetState()

    def resetState(self):
        self.s = np.array([self.start_x, self.start_v])
        self.r_cum = 0.0

    def takeAction(self, a_idx: int) -> float:
        a = self.actions[a_idx]
        self.s[1] += 0.001*a - 0.0025*math.cos(3*self.s[0])

        if self.s[1] < self.v_lb:
            self.s[1] = self.v_lb
        if self.s[1] > self.v_ub:
            self.s[1] = self.v_ub

        self.s[0] += self.s[1]

        # Check for collision conditions
        if self.s[0] < self.x_lb:
            self.s[0] = self.x_lb
            self.s[1] = 0.0
        if self.s[0] > self.x_ub:
            self.s[0] = self.x_ub
            self.s[1] = 0.0

        if np.all(self.s == self.goal_state):
            r = 0.0
        else:
            r = -1.0
        self.r_cum += r

        if self.r_cum == -50000:
            self.s = self.goal_state

        return r

    def returnFourierBasis(self, order: int) -> np.ndarray:
        # normalize state to [0,1] based on lb/ub's
        norm_s = self.s.copy()
        norm_s[0] = (norm_s[0] - self.x_lb)/(self.x_ub - self.x_lb)
        norm_s[1] = (norm_s[1] - self.v_lb)/(self.v_ub - self.v_lb)

        # calculate phi without pi and cos across all combinations up to order
        c = np.array(np.meshgrid(*[[i for i in range(order+1)] for j in range(norm_s.size)])).T.reshape(-1, norm_s.size)
        
        return np.cos(np.pi*np.dot(c, norm_s))
