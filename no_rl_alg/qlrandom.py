from random import randint
from qlrmalgo import QLRMAlgo

# ==========================================================================================================
# Agent who acts randomly
# ==========================================================================================================
class QLRandom(QLRMAlgo):
	def __init__(self, name, game_model, **kwargs):
		super().__init__(name, game_model, **kwargs)

	def get_best_action(self, state, rm_belief):
			return randint(0,4)

	def initialize_q_table(self):
			return {}

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
			 return 0, 0

	def get_q_value(self, state, rm_belief, action):
			return 0