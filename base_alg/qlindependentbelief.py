from qlrmalgo import QLRMAlgo
from random import random
import numpy as np


# ==========================================================================================================
# QL with probabilistic updates to the RM belief.
# When the `decorrelate` argument is set to true, the belief is only updated the first time the agent digs
# at one of the relevant squares.
# ==========================================================================================================
class QLIndependentBelief(QLRMAlgo):
  def __init__(self, name, game_model, decorrelate = False, movement_cost=0.02, **kwargs):
    super().__init__(name, game_model, **kwargs)
    self.decorrelate = decorrelate
    self.movement_cost = movement_cost

  def initialize_q_table(self):
    board_lenght = self.get_game_model().get_board_lenght()
    depot = self.get_game_model().get_depot()
    action_lenght = self.get_action_space_lenght()    
    dug_lenght = self.get_dug_lenght()
      
    self.q1 = { rm_state : 0 if rm_state == 2 else (random() - 0.5) for rm_state in range(3) }
    self.q2 = { (rm_state, pos, a) : 0 if rm_state == 2 or pos == depot else (np.random.random() - 0.5) for rm_state in range(3) for pos in range(board_lenght) for a in range(action_lenght) }
    self.q3 = { (rm_state, pos, dug_pos, a) : 0 if rm_state == 2 or pos == depot else (np.random.random() - 0.5) for rm_state in range(3) for pos in range(board_lenght) for dug_pos in range(dug_lenght) for a in range(action_lenght) }

  def get_q_value(self, state, rm_belief, action):
    dug_lenght = self.get_dug_lenght()
      
    q_sum = 0
    for u in range(3):
      q_sum += self.q1[u] * rm_belief[0][u]
      q_sum += self.q2[(u, state[0], action)] * rm_belief[0][u]
      for dug_pos in range(dug_lenght):
        q_sum += self.q3[(u, state[0], dug_pos, action)] * rm_belief[0][u] * state[2][dug_pos] / dug_lenght
    return q_sum

  def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
    dug_lenght = self.get_dug_lenght()      
      
    if done:
      delta = reward - self.get_q_value(state, rm_belief, action)
    else:
      delta = reward + self.discount * self.get_state_value(next_state, next_rm_belief) - self.get_q_value(state, rm_belief, action)

    for u in range(3):
      self.q1[u] += self.lr * delta * rm_belief[0][u]
      self.q2[(u, state[0], action)] += self.lr * delta * rm_belief[0][u]

      for dug_pos in range(dug_lenght):
        self.q3[(u, state[0], dug_pos, action)] += self.lr * delta * rm_belief[0][u] * state[2][dug_pos] / dug_lenght

  def initialize_rm_belief(self):
    board_lenght = self.get_game_model().get_board_lenght()      
    dug = [False] * board_lenght
    rm_belief = np.array((1,0,0))
    return rm_belief, dug

  def update_rm_belief(self, rm_belief, state, action, next_state):
    depot = self.get_game_model().get_depot()      
    has_gold_model = self.get_game_model().get_has_gold_model()
    dig_action = self.get_game_model().get_dig_action()
    
    next_pos = next_state[0]
    rm_belief, dug = rm_belief
    predicted_reward = 0

    if next_pos == depot:
      next_rm_belief = np.array((rm_belief[0], 0, rm_belief[1]))
      predicted_reward = rm_belief[1]

    elif action == dig_action:
      if self.decorrelate and dug[next_pos]:
        p1 = rm_belief[1]
      else:
        p1 = rm_belief[1] + rm_belief[0] * has_gold_model[next_pos]
      next_rm_belief = np.array((1-p1, p1, 0))
      dug[next_pos] = True
    else:
      next_rm_belief = rm_belief

    if action in [0,1,2,3]:
      predicted_reward -= self.movement_cost

    return (next_rm_belief, dug), predicted_reward