from random import *
import numpy as np

# A simple toy gridworld environment to test Reward Machine RL algorithms with noisy detectors.
# ==========================================================================================================
# MAP: (S = start, D = depot, G = gold, F = fool's gold)
# S . . G
# . F . G
# . F . G
# D . . G
# ==========================================================================================================
# Objective:  Mine at least one ore of gold and deposit it at the depot. However, the agent cannot
# accurately distinguish real gold from fool's gold, and assigns the following probabilities in its belief
# of whether there is gold at each square.
#
# 0.0  0.0  0.0  0.8
# 0.0  0.3  0.0  0.8
# 0.0  0.6  0.0  0.8
# 0.0  0.0  0.0  0.8
#
# ==========================================================================================================
# States: integers [0,15], representing the agent's location in the grid. The top left is 0, top right is 3,
# bottom left is 12, bottom right is 15.
#
# Actions: integers [0,4]. (0 = left,1 = right,2 = up,3 = down, 4 = dig).
#
# Reward Machine (task) state:
# 0 = haven't acquired gold
# 1 = acquired gold, but haven't deposited
# 2 = deposited gold


class MiningEnv:
  depot = 12

  has_gold = [False, False, False, True,
              False, False, False, True,
              False, False, False, True,
              False, False, False, True]

  has_gold_model = [0, 0, 0, 0.8,
                    0, 0.3, 0, 0.8,
                    0, 0.6, 0, 0.8,
                    0, 0., 0, 0.8]
  relevant_squares = {3:0, 5:1, 7:2, 11:3, 13:4, 15:5}

  LEFT = 0
  RIGHT = 1
  UP = 2
  DOWN = 3
  DIG = 4
  
  agents_initial_pos = [0, 4, 8, 12]

  def __init__(self, agent_names, max_steps=500, movement_cost=0.02, show_board=False, collision=True):
    self.max_steps = max_steps
    self.movement_cost = movement_cost
    self.agent_names = agent_names
    
    self.pos = { self.agent_names[i]: self.agents_initial_pos[i] for i in range(len(self.agent_names)) }
    self.rm_state = { name: 0 for name in self.agent_names }
    self.visited = { name: np.array([False for i in range(6)]) for name in self.agent_names }
    self.steps = 0

    self.show_board = show_board
    self.collision = collision
    
  def is_occupied(self, pos):
    if self.collision:
      for name in self.agent_names:
        if self.pos[name] == pos:
          return True
    return False

  def move_agent(self, agent_name, walk):
    new_pos = self.pos[agent_name] + walk

    if self.is_occupied(new_pos):
      #print(f'Colission at {new_pos}!')
      return

    self.pos[agent_name] = new_pos


  def reset(self, agent_name):
    self.pos[agent_name] = 0
    self.rm_state[agent_name] = 0
    self.visited[agent_name] = np.array([False for i in range(6)])
    self.steps = 0

    pos = self.pos[agent_name]
    rm_state = self.rm_state[agent_name]
    visited = np.copy(self.visited[agent_name])
    other_pos = self.get_other_agent_pos(agent_name)

    return (pos, rm_state, visited, other_pos)

  def show_agents_position(self):
    print()

    board = ['-' for i in range(16)]

    for i in range(16):
      if self.has_gold[i]:
        board[i] = 'g'

    for name in self.agent_names:
      board[self.pos[name]] = name

    for i in range(4):
      print(f'{i:}\t', end='')
      for j in range(4):
        print(f'\t{board[i * 4 + j]}', end='')
      print()

    print()
    
  def get_other_agent_pos(self, agent_name):
    for name in self.agent_names:
        if name != agent_name:
            return self.pos[name]
    return 0
        

  def step(self, agent_name, action):
    self.steps += 1
    reward = 0
    done = False

    if action in [0,1,2,3]:
      reward -= self.movement_cost

    # Left
    if action == self.LEFT:
      if self.pos[agent_name] % 4 != 0:
        self.move_agent(agent_name, -1)
    # Right
    elif action == self.RIGHT:
      if self.pos[agent_name] % 4 != 3:
        self.move_agent(agent_name, 1)
    # Up
    elif action == self.UP:
      if self.pos[agent_name] < 12:
        self.move_agent(agent_name, 4)
    # Down
    elif action == self.DOWN:
      if self.pos[agent_name] > 3:
        self.move_agent(agent_name, -4)
    # Dig
    elif action == self.DIG:
      if self.has_gold[self.pos[agent_name]] and self.rm_state[agent_name] == 0:
        self.rm_state[agent_name] = 1
      if self.pos[agent_name] in self.relevant_squares:
        self.visited[agent_name][self.relevant_squares[self.pos[agent_name]]] = True

    # Check if we're at the storage
    if self.pos[agent_name] == self.depot:
      if self.rm_state[agent_name] == 1:
          self.rm_state[agent_name] = 2
          reward += 1
          done = True

    if self.steps == self.max_steps:
      done = True

    pos = self.pos[agent_name]
    rm_state = self.rm_state[agent_name]
    visited = np.copy(self.visited[agent_name])
    other_pos = self.get_other_agent_pos(agent_name)

    if self.show_board:
      self.show_agents_position()

    return (pos, rm_state, visited, other_pos), reward, done, None