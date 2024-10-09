from miningenv import MiningEnv
from random import random, randint

class NoAdGameControl:
  def __init__(
    self,
    agent,
	reward_type="actual",	# ["actual", "predicted"]
	discount=0.99, 			# Discount factor γ
	eps=0.2, 					# Random action probability
	lr=0.01,					# Learning rate
	max_frames=1e6,
	log_interval=1e4,
  ):
    self.agent = agent
    self.reward_type = reward_type
    self.discount = discount
    self.eps = eps
    self.max_frames= max_frames
    self.log_interval = log_interval

     # Initialize envs and logs
    self.agent.initialize_q_table()
    self.logs = {'frames': [], 'return': [], 'discounted_return': [], 'predicted_return': [], 'predicted_discounted_return': []}

  def create_env(self, agent):
    env = MiningEnv([agent.get_name(), ])
    return env


  # Train the RM algorithm up to `max_frames` frames.
  def train(self, agent, print_logs=False):
    eps_num = 0
    frames = 0

    env = self.create_env(agent)

    while frames < self.max_frames:
      state = env.reset(agent.get_name())
      rm_belief = agent.initialize_rm_belief()
      eps_num += 1
      eps_len = 0

      # Simulate one episode
      while True:
        # Action selection
        if random() < self.eps:
          action = randint(0,4)
        else:
          action = agent.get_best_action(state, rm_belief)

        # Step environment
        next_state, reward, done, _ = env.step(agent.get_name(), action)

        # Update RM belief
        next_rm_belief, predicted_reward = agent.update_rm_belief(rm_belief, state, action, next_state)
        if self.reward_type == "predicted":
          if predicted_reward is None:
            raise RuntimeError("Predicted reward is not defined.")
          reward = predicted_reward

        # Update Q-values
        agent.update_q_values(state, rm_belief, action, reward, next_state, next_rm_belief, done)

        state = next_state
        rm_belief = next_rm_belief
        eps_len += 1

        # Log results periodically
        if (frames + eps_len) % self.log_interval == 0:
          self.log_policy(agent, frames + eps_len, print_logs)

        if done:
          frames += eps_len
          break

  # Evaluate the current policy over some number of episodes, and log the results
  # in self.logs
  def log_policy(self, agent, frames, print_logs=False):
    returnn, discounted_returnn, predicted_returnn, predicted_discounted_returnn = self.eval_episode(agent)
    self.logs['frames'].append(frames)
    self.logs['return'].append(returnn)
    self.logs['discounted_return'].append(discounted_returnn)
    self.logs['predicted_return'].append(predicted_returnn)
    self.logs['predicted_discounted_return'].append(predicted_discounted_returnn)

    if print_logs:
      print("Frames: %.2f, R: %.2f, Disc R: %.2f -- Pred R: %.2f, Disc Pred R: %.2f"%(frames, returnn, discounted_returnn, predicted_returnn, predicted_discounted_returnn))


  # Evaluate one episode of the current policy.
  def eval_episode(self, agent):
    env_eval = self.create_env(agent)

    state = env_eval.reset(agent.get_name())
    rm_belief = agent.initialize_rm_belief()

    eps_len = 0
    returnn = 0
    discounted_returnn = 0
    predicted_returnn = 0
    predicted_discounted_returnn = 0

    while True:
      action = agent.get_best_action(state, rm_belief)
      next_state, reward, done, _ = env_eval.step(agent.get_name(), action)
      next_rm_belief, predicted_reward = agent.update_rm_belief(rm_belief, state, action, next_state)

      state = next_state
      rm_belief = next_rm_belief

      eps_len += 1
      returnn += reward
      discounted_returnn += reward * self.discount ** (eps_len - 1)

      if predicted_reward is not None:
        predicted_returnn += predicted_reward
        predicted_discounted_returnn += predicted_reward * self.discount ** (eps_len - 1)

      if done or eps_len == 100:
        break
    return returnn, discounted_returnn, predicted_returnn, predicted_discounted_returnn


  def match(self):
    env = self.create_env(self.agent)

    state = env.reset(self.agent.get_name())

    rm_belief = self.agent.initialize_rm_belief()


    for i in range(100):
        print(f'Step: {i}')
        env.show_agents_position()

        action = self.agent.get_best_action(state, rm_belief)
        state, reward, done, _ = env.step(self.agent.get_name(), action)
        
        print(f'State 1: {state}')
        
        if done:
          break
      
        next_rm_belief, predicted_reward = self.agent.update_rm_belief(rm_belief, state, action, state)
        rm_belief = next_rm_belief        

    return self.agent