from qlperfectrm import QLPerfectRM
from qlperfectrm2 import QLPerfectRM2
from qlnorm import QLNoRM
from qlnorm2 import QLNoRM2
from qlindependentbelief import QLIndependentBelief
from qlbeliefthresholding import QLBeliefThresholding
from qlbeliefthresholding2 import QLBeliefThresholding2
from mp_gamecontrol import MultiPlayersGameControl

import pickle


agents = [
    QLNoRM('QLNorm'),
    QLPerfectRM('QLPerfectRM'),
    QLBeliefThresholding("QLBeliefThresholding", movement_cost=0.02),
    QLIndependentBelief('QLIndependentBelief_F', decorrelate=False),
    QLIndependentBelief('QLIndependentBelief_T', decorrelate=True)    
]


for agent in agents:
    
    my_data = {}    
    
    for i in range(10):
        print(f'{agent.get_name()} - Execution {i}...')
        
        control = MultiPlayersGameControl([agent], max_frames=5e6, log_interval=10000, save_agents_models=(i == 0), models_path="c:\\temp\\aamas2025\\models")
        control.train(agent, print_logs=True)
        
        frames = control.logs['frames']
        returns = control.logs['return']
        
        for j in range(len(frames)):
            if not frames[j] in my_data:
                my_data[frames[j]] = []
            my_data[frames[j]].append(returns[j]) 
    
    
    print()
    print('Saving...')
    
    with open(f'C:\\temp\\aamas2025\\data_{agent.get_name()}.pck', 'wb') as my_file:
        pickle.dump(my_data, my_file, protocol=pickle.HIGHEST_PROTOCOL)    


print('Finished!')