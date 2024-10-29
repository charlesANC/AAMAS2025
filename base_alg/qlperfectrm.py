from qlrmalgo import QLRMAlgo
from random import random

# ==========================================================================================================
# QL with the perfect RM state
# ==========================================================================================================
class QLPerfectRM(QLRMAlgo):
	def __init__(self, name, game_model, **kwargs):
		super().__init__(name, game_model, **kwargs)

	def initialize_q_table(self):
		depot = self.get_depot()
		lenght = self.get_board_lenght()    
		action_lenght = self.get_action_space_lenght()
    
        
		self.q = { (pos, rm_state, a): 0 if rm_state == 2 or pos == depot else (random() - 0.5)
                   for pos in range(lenght)                        
						for rm_state in range(3)
						for a in range(action_lenght)}

	def get_q_value(self, state, rm_belief, action):
		return self.q[(state[0], state[1], action)]

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			self.q[(state[0], state[1], action)] += self.lr * (reward - self.q[(state[0], state[1], action)])
		else:
			self.q[(state[0], state[1], action)] += self.lr * (reward + self.discount * self.get_state_value(next_state, None) - self.q[(state[0], state[1], action)])