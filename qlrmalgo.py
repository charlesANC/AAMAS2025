# Implements simple tabular Q-learning algorithms for the gold mining domain, using linear approximations for generalization.
# Memory is implemented by conditioning on a set of relevant history features: namely, whether the agent has visited each of
# the 6 squares containing either gold or fool's gold.
# You can train the agent in one of two modes:
# - using ground-truth rewards
# - using the agent's predicted rewards


# ==========================================================================================================
# ==========================================================================================================
# Base class implementing Q-learning RM algorithms
# ==========================================================================================================
# ==========================================================================================================


class QLRMAlgo:
	def __init__(self,
		name,                 # agent's unique name
		game_model,           # some information about the game    
		reward_type="actual",	# ["actual", "predicted"]
		discount=0.99, 				# Discount factor γ
		eps=0.2, 					# Random action probability
		lr=0.01,					# Learning rate
	):
		self.name = name
		self.game_model = game_model

		# RL Hyperparameters
		self.reward_type = reward_type
		self.discount = discount
		self.eps = eps
		self.lr = lr

		# Initialize envs and logs
		self.initialize_q_table()

	# ==========================================================================================================
	# Internal methods
	# ==========================================================================================================

	def get_name(self):
		return self.name
    
	def get_game_model(self):
		return self.game_model
    
	def get_best_action(self, state, rm_belief):
		best_action = None
		best_q = -99999999999

		for action in range(5):
			qsa = self.get_q_value(state, rm_belief, action)
			if qsa > best_q:
				best_action = action
				best_q = qsa

		return best_action

	def get_state_value(self, state, rm_belief):
		best_value = -99999999999

		for a in range(5):
			best_value = max(best_value, self.get_q_value(state, rm_belief, a))

		return best_value

	# ==========================================================================================================
	# Protocol methods (you need to define these in the subclass)
	# ==========================================================================================================

	# Initializes the Q-value table.
	def initialize_q_table(self):
		raise NotImplementedError()

	# Predict the Q value of the given state-action pair using the values in self.q.
	def get_q_value(self, state, rm_belief, action):
		raise NotImplementedError()

	# Given an experience, update the Q-table.
	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		raise NotImplementedError()

	# Return the initial RM belief
	def initialize_rm_belief(self):
		return None

	# Returns: (next_rm_belief, predicted_reward)
	# Override this in the subclass if needed.
	def update_rm_belief(self, rm_belief, state, action, next_state):
		return rm_belief, None
    
   # Methods created to allow agents to work on
   # more than one environment configuration
            
	def get_board_lenght(self):
		return self.get_game_model().get_board_lenght()
            
	def get_action_space_lenght(self):
		return self.get_game_model().get_action_space_lenght()

	def get_depot(self):
		return self.get_game_model().get_depot()
    
	def get_dug_lenght(self):
		return self.get_game_model().get_dug_lenght()    