from ext_alg.qlindependentbelief2 import QLIndependentBelief2
from random import random, randint


# ==========================================================================================================
# Same as QLIndependentBelief2 but take random actions
# ==========================================================================================================
class QLIndependentBelief2R(QLIndependentBelief2):
  def __init__(self, name, game_model, decorrelate = False, movement_cost=0.02, eps=0.2, **kwargs):
    super().__init__(name, game_model, decorrelate, movement_cost, **kwargs)
    self.eps = eps
    

  def get_best_action(self, state, rm_belief):
    if random() < self.eps:
        action = randint(0,4)
    else:
        action = super().get_best_action(state, rm_belief)    
    return action


