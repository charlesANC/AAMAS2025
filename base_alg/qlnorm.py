from qlrmalgo import QLRMAlgo
from random import random

# ==========================================================================================================
# QL with memory only (no RM)
# ==========================================================================================================
class QLNoRM(QLRMAlgo):
	def __init__(self, name, game_model, **kwargs):
		super().__init__(name, game_model, **kwargs)

	def initialize_q_table(self):
		depot = self.get_depot()
		dug_len = self.get_dug_lenght()
		board_length = self.get_board_lenght()
		action_len = self.get_action_space_lenght()
        
		self.q1 = { (pos, a): 0 if pos == depot else (random() - 0.5) for pos in range(board_length) for a in range(action_len)  }
		self.q2 = { (pos, dug_pos, a): 0 if pos == depot else (random() - 0.5) for pos in range(board_length) for dug_pos in range(dug_len) for a in range(action_len)  }

	def get_q_value(self, state, rm_belief, action):
		dug_len = self.get_dug_lenght()
        
		q_sum = 0
		q_sum += self.q1[(state[0], action)]
		for dug_pos in range(dug_len):
			q_sum += self.q2[(state[0], dug_pos, action)] * state[2][dug_pos] / dug_len
		return q_sum

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		dug_len = self.get_dug_lenght()    
        
		if done:
			delta = reward - self.get_q_value(state, None, action)
		else:
			delta = reward + self.discount * self.get_state_value(next_state, None) - self.get_q_value(state, None, action)

		self.q1[(state[0], action)] += self.lr * delta
		for dug_pos in range(dug_len):
			self.q2[(state[0], dug_pos, action)] += self.lr * delta * state[2][dug_pos] / dug_len
