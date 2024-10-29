from qlrmalgo import QLRMAlgo
from random import random

# ==========================================================================================================
# QL with belief thresholding
# ==========================================================================================================
class QLBeliefThresholding(QLRMAlgo):
  def __init__(self, name, game_model, movement_cost, **kwargs):
    super().__init__(name, game_model, **kwargs)
    self.movement_cost = movement_cost

  def initialize_q_table(self):
    board_lenght = self.get_game_model().get_board_lenght()
    depot = self.get_game_model().get_depot()
    action_lenght = self.get_action_space_lenght()    
    dug_lenght = self.get_dug_lenght()
    
    self.q1 = { rm_state : 0 if rm_state == 2 else (random() - 0.5) for rm_state in range(3) }
    self.q2 = { (rm_state, pos, a) : 0 if rm_state == 2 or pos == depot else (random() - 0.5) for rm_state in range(3) for pos in range(board_lenght) for a in range(action_lenght) }
    self.q3 = { (rm_state, pos, dug_pos, a) : 0 if rm_state == 2 or pos == depot else (random() - 0.5) for rm_state in range(3) for pos in range(board_lenght) for dug_pos in range(dug_lenght) for a in range(action_lenght) }

  def get_q_value(self, state, rm_belief, action):
    dug_lenght = self.get_dug_lenght()
      
    q_sum = 0

    q_sum += self.q1[rm_belief] + self.q2[(rm_belief, state[0], action)]
    for dug_pos in range(dug_lenght):
      q_sum += self.q3[(rm_belief, state[0], dug_pos, action)] * state[2][dug_pos] / dug_lenght

    return q_sum

  def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
    dug_lenght = self.get_dug_lenght()
      
    if done:
      delta = reward - self.get_q_value(state, rm_belief, action)
    else:
      delta = reward + self.discount * self.get_state_value(next_state, next_rm_belief) - self.get_q_value(state, rm_belief, action)

    self.q1[rm_belief] += self.lr * delta
    self.q2[(rm_belief, state[0], action)] += self.lr * delta

    for dug_pos in range(dug_lenght):
      self.q3[(rm_belief, state[0], dug_pos, action)] += self.lr * delta * state[2][dug_pos] / dug_lenght

  def initialize_rm_belief(self):
    return 0

  def update_rm_belief(self, rm_belief, state, action, next_state):
    depot = self.get_game_model().get_depot()
    dig_action = self.get_game_model().get_dig_action
    has_gold_model = self.get_game_model().get_has_gold_model()
    
    predicted_reward = 0

    next_rm_belief = rm_belief
    if rm_belief == 0 and action == dig_action and has_gold_model[next_state[0]] >= 0.5:
      next_rm_belief = 1
    elif rm_belief == 1 and next_state[0] == depot:
      next_rm_belief = 2
      predicted_reward = 1

    if action in [0,1,2,3]:
      predicted_reward -= self.movement_cost

    return next_rm_belief, predicted_reward