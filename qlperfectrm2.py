from qlrmalgo import QLRMAlgo
from miningenv import MiningEnv
from random import random

# ==========================================================================================================
# QL with the perfect RM state
# ==========================================================================================================
class QLPerfectRM2(QLRMAlgo):
	def __init__(self, name, **kwargs):
		super().__init__(name, **kwargs)

	def initialize_q_table(self):
		self.q = { (pos, pos_op, rm_state, a): 0 if rm_state == 2 or pos == MiningEnv.depot else (random() - 0.5)
						for pos in range(16)
						for pos_op in range(16)
						for rm_state in range(3)
						for a in range(5)}

	def get_q_value(self, state, rm_belief, action):
		return self.q[(state[0], state[3], state[1], action)]

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			self.q[(state[0], state[3], state[1], action)] += self.lr * (reward - self.q[(state[0], state[3], state[1], action)])
		else:
			self.q[(state[0], state[3], state[1], action)] += self.lr * (reward + self.discount * self.get_state_value(next_state, None) - self.q[(state[0], state[3], state[1], action)])