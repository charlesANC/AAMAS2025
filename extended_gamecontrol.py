from extended_miningenv import ExtMiningEnv
from no_rl_alg.qlrandom import QLRandom
from random import random, randint
import pickle

class ExtGameControl:
  def __init__(
    self,
    agent1,
    agent2,
		reward_type="actual",	# ["actual", "predicted"]
		discount=0.99, 			# Discount factor γ
		eps=0.2, 					# Random action probability
		lr=0.01,					# Learning rate
		max_frames=1e6,
		log_interval=1e4,
		save_agents_models=False,
		models_path="c:\\temp\\"        
  ):
    self.agent1 = agent1
    self.agent2 = agent2
    self.reward_type = reward_type
    self.discount = discount
    self.eps = eps
    self.max_frames= max_frames
    self.log_interval = log_interval

     # Initialize envs and logs
    self.agent1.initialize_q_table()
    self.agent2.initialize_q_table()
    self.logs = {'frames': [], 'return': [], 'discounted_return': [], 'predicted_return': [], 'predicted_discounted_return': []}
    
    self.save_agents_models = save_agents_models
    self.models_path = models_path    

  def create_env_for_training(self, agent):
    random_agent = QLRandom('R', ExtMiningEnv.game_model)
    env = ExtMiningEnv([agent.get_name(), random_agent.get_name()])
    return random_agent, env

  def create_env_for_duel(self, agent1, agent2):
    env = ExtMiningEnv([agent1.get_name(), agent2.get_name()], max_steps=5000)
    return env

  # Train the RM algorithm up to `max_frames` frames.
  def train(self, agent, print_logs=False):
    eps_num = 0
    frames = 0
    
    max_return = -999    

    opponent, env = self.create_env_for_training(agent)

    while frames < self.max_frames:
      env.reset(opponent.get_name())

      state = env.reset(agent.get_name())
      rm_belief = agent.initialize_rm_belief()
      eps_num += 1
      eps_len = 0

      # Simulate one episode
      while True:
        opp_action = opponent.get_best_action(None, None)
        env.step(opponent.get_name(), opp_action)

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
          this_return = self.log_policy(agent, frames + eps_len, print_logs)
          if this_return >= max_return:
              max_return = this_return
              if self.save_agents_models:
                  print(f'Saving {agent.get_name()} model at {frames + eps_len}...')
                  self.save_agent_model(agent)          

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
      print("Agent: %s -- Frames: %.2f, R: %.2f, Disc R: %.2f -- Pred R: %.2f, Disc Pred R: %.2f"%(agent.get_name(), 
            frames, returnn, discounted_returnn, predicted_returnn, 
                predicted_discounted_returnn))
      
    return returnn


  # Evaluate one episode of the current policy.
  def eval_episode(self, agent):
    opponent, env_eval = self.create_env_for_training(agent)

    env_eval.reset(opponent.get_name())
    state = env_eval.reset(agent.get_name())
    rm_belief = agent.initialize_rm_belief()

    eps_len = 0
    returnn = 0
    discounted_returnn = 0
    predicted_returnn = 0
    predicted_discounted_returnn = 0

    while True:
      opp_action = opponent.get_best_action(None, None)
      env_eval.step(opponent.get_name(), opp_action)

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


  def match(self, inverse=False):
    env = self.create_env_for_duel(self.agent1, self.agent2)
    
    if inverse:
        agent1 = self.agent2
        agent2 = self.agent1
    else:
        agent1 = self.agent1
        agent2 = self.agent2

    state_1 = env.reset(agent1.get_name())
    state_2 = env.reset(agent2.get_name())

    rm_belief_1 = agent1.initialize_rm_belief()
    rm_belief_2 = agent2.initialize_rm_belief()

    for i in range(10000):
        print(f'Step: {i}')
        env.show_agents_position()

        action = agent1.get_best_action(state_1, rm_belief_1)
        next_state_1, reward, done, _ = env.step(agent1.get_name(), action)
        print(f'State 1: {state_1}')
        rm_belief_1, predicted_reward = agent1.update_rm_belief(rm_belief_1, state_1, action, next_state_1)        
        state_1 = next_state_1

        if done:
          break

        action = agent2.get_best_action(state_2, rm_belief_2)
        next_state_2, reward, done, _ = env.step(agent2.get_name(), action)
        print(f'State 2: {state_2}')
        rm_belief_2, predicted_reward = agent2.update_rm_belief(rm_belief_2, state_2, action, next_state_2)        
        state_2 = next_state_2

        if done:
          break

    if state_1[1] == 2 and state_2[1] != 2:
      return agent1
    elif state_2[1] == 2 and state_1[1] != 2:
      return agent2

    return None


  def save_agent_model(self, agent):
      with open(f'{self.models_path}\\{agent.get_name()}_model.pck', 'wb') as model_file:
          pickle.dump(agent, model_file, protocol=pickle.HIGHEST_PROTOCOL) 