from qlperfectrm import QLPerfectRM
from qlperfectrm2 import QLPerfectRM2
from qlnorm import QLNoRM
from qlnorm2 import QLNoRM2
from qlindependentbelief import QLIndependentBelief
from qlindependentbelief2 import QLIndependentBelief2
from qlbeliefthresholding import QLBeliefThresholding
from qlbeliefthresholding2 import QLBeliefThresholding2
from noad_gamecontrol import NoAdGameControl

import pickle


file_name = "QLBeliefThresholding2"

#agent1 = QLPerfectRM2('A')
#agent1 = QLNoRM('A')
#agent1 = QLBeliefThresholding('A', movement_cost=0.02)
#agent1 = QLIndependentBelief('A', decorrelate=False)
#agent1 = QLIndependentBelief('A', decorrelate=True)


my_data = {}


for i in range(10):
    print(f'Execution {i}...')

    agent1 = QLPerfectRM('A')
    control = NoAdGameControl(agent1, max_frames=1e6, log_interval=10000)
    control.train(agent1, print_logs=True)
    
    frames = control.logs['frames']
    returns = control.logs['return']
    
    for j in range(len(frames)):
        if not frames[j] in my_data:
            my_data[frames[j]] = []
        my_data[frames[j]].append(returns[j]) 


print()
print('Saving...')

with open(f'C:\\temp\\aamas2025\\{file_name}.pck', 'wb') as my_file:
    pickle.dump(my_data, my_file, protocol=pickle.HIGHEST_PROTOCOL)    


print('Finished!')