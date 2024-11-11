from base_alg.qlindependentbelief import QLIndependentBelief
from ext_alg.qlindependentbelief2 import QLIndependentBelief2
from extended_gamecontrol import ExtGameControl
from extended_miningenv import ExtMiningEnv
import random

import pickle

seed = 443198418283
random.seed(seed)
    
my_data = {}    

for i in range(1):
    
    agent1 = QLIndependentBelief('QLIndependentBelief_TM__SEED', ExtMiningEnv.game_model, decorrelate=True)
    agent2 = QLIndependentBelief2('QLIndependentBelief2_TM', ExtMiningEnv.game_model, decorrelate=True)    
    
    print(f'{agent1.get_name()} - Execution {i}...')
    
    
    control = ExtGameControl(agent1, agent2, max_frames=2e6, log_interval=10000, save_agents_models=True, models_path="c:\\temp\\aamas2025\\models")
    #        control = MultiPlayersGameControl([agent], max_frames=20e6, log_interval=10000, save_agents_models=False, models_path="c:\\temp\\aamas2025\\models")
    control.train(agent1, print_logs=True)
    
    frames = control.logs['frames']
    returns = control.logs['return']
    
    for j in range(len(frames)):
        if not frames[j] in my_data:
            my_data[frames[j]] = []
        my_data[frames[j]].append(returns[j]) 


print()
print('Saving...')

with open(f'C:\\temp\\aamas2025\\data_{agent1.get_name()}.pck', 'wb') as my_file:
    pickle.dump(my_data, my_file, protocol=pickle.HIGHEST_PROTOCOL)    


print('Finished!')