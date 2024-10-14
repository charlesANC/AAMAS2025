from qlrmalgo import QLRMAlgo
from miningenv import MiningEnv
from random import random

# ==========================================================================================================
# QL with memory only (no RM)
# ==========================================================================================================
class QLNoRM2(QLRMAlgo):
	def __init__(self, name, **kwargs):
		super().__init__(name, **kwargs)

	def initialize_q_table(self):
		self.q1 = { (pos, pos_op, a): 0 if pos == MiningEnv.depot else (random() - 0.5) 
             for pos in range(16) 
             for pos_op in range(16)              
             for a in range(5)  }
		self.q2 = { (pos, dug_pos, a): 0 if pos == MiningEnv.depot else (random() - 0.5) for pos in range(16) for dug_pos in range(6) for a in range(5)  }

	def get_q_value(self, state, rm_belief, action):
		q_sum = 0
		q_sum += self.q1[(state[0], state[3], action)]
		for dug_pos in range(6):
			q_sum += self.q2[(state[0], dug_pos, action)] * state[2][dug_pos] / 6
		return q_sum

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			delta = reward - self.get_q_value(state, None, action)
		else:
			delta = reward + self.discount * self.get_state_value(next_state, None) - self.get_q_value(state, None, action)

		self.q1[(state[0], state[3], action)] += self.lr * delta
		for dug_pos in range(6):
			self.q2[(state[0], dug_pos, action)] += self.lr * delta * state[2][dug_pos] / 6