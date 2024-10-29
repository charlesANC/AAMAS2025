from qlrmalgo import QLRMAlgo
from miningenv import MiningEnv
from random import random

# ==========================================================================================================
# QL with belief thresholding
# ==========================================================================================================
class QLBeliefThresholding2(QLRMAlgo):
  def __init__(self, name, movement_cost, **kwargs):
    super().__init__(name, **kwargs)
    self.movement_cost = movement_cost

  def initialize_q_table(self):
    self.q1 = { rm_state : 0 if rm_state == 2 else (random() - 0.5) for rm_state in range(3) }
    
    self.q2 = { (rm_state, pos, pos_op, a) : 0 if rm_state == 2 or pos == MiningEnv.depot else (random() - 0.5) 
               for rm_state in range(3) 
               for pos in range(16) 
               for pos_op in range(16)
               for a in range(5) }
    
    self.q3 = { (rm_state, pos, dug_pos, a) : 0 if rm_state == 2 or pos == MiningEnv.depot else (random() - 0.5) for rm_state in range(3) for pos in range(16) for dug_pos in range(6) for a in range(5) }

  def get_q_value(self, state, rm_belief, action):
    q_sum = 0

    q_sum += self.q1[rm_belief] + self.q2[(rm_belief, state[0], state[3], action)]
    for dug_pos in range(6):
      q_sum += self.q3[(rm_belief, state[0], dug_pos, action)] * state[2][dug_pos] / 6

    return q_sum

  def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
    if done:
      delta = reward - self.get_q_value(state, rm_belief, action)
    else:
      delta = reward + self.discount * self.get_state_value(next_state, next_rm_belief) - self.get_q_value(state, rm_belief, action)

    self.q1[rm_belief] += self.lr * delta
    self.q2[(rm_belief, state[0], state[3], action)] += self.lr * delta

    for dug_pos in range(6):
      self.q3[(rm_belief, state[0], dug_pos, action)] += self.lr * delta * state[2][dug_pos] / 6

  def initialize_rm_belief(self):
    return 0

  def update_rm_belief(self, rm_belief, state, action, next_state):
    predicted_reward = 0

    next_rm_belief = rm_belief
    if rm_belief == 0 and action == MiningEnv.DIG and MiningEnv.has_gold_model[next_state[0]] >= 0.5:
      next_rm_belief = 1
    elif rm_belief == 1 and next_state[0] == MiningEnv.depot:
      next_rm_belief = 2
      predicted_reward = 1

    if action in [0,1,2,3]:
      predicted_reward -= self.movement_cost

    return next_rm_belief, predicted_reward