from qlperfectrm import QLPerfectRM
from qlperfectrm2 import QLPerfectRM2
from qlnorm import QLNoRM
from qlindependentbelief import QLIndependentBelief
from qlbeliefthresholding import QLBeliefThresholding
from noad_gamecontrol import NoAdGameControl

import matplotlib.pyplot as plt
import numpy as np



#agent1 = QLPerfectRM('A')
#agent1 = QLPerfectRM2('A')
#agent1 = QLNoRM('A')
#agent1 = QLBeliefThresholding('A', movement_cost=0.02)
agent1 = QLIndependentBelief('A', decorrelate=False)
#agent1 = QLIndependentBelief('A', decorrelate=True)



control = NoAdGameControl(agent1, max_frames=1e6, log_interval=10000)

control.train(agent1, print_logs=True)


# define data values

plt.plot(control.logs['frames'], control.logs['return'])  # Plot the chart
plt.show()  # display



print()
print()


#state = [11, 0, [True, True, True, True, False, False], 10]

#print(f'for state {state}: {agent1.get_q_value(state, 0, 0)}')
#print(f'for state {state}: {agent1.get_q_value(state, 0, 1)}')
#print(f'for state {state}: {agent1.get_q_value(state, 0, 2)}')
#print(f'for state {state}: {agent1.get_q_value(state, 0, 3)}')
#print(f'for state {state}: {agent1.get_q_value(state, 0, 4)}')


print()
print()


#state = [11, 1, [True, True, True, True, False, False], 10]

#print(f'for state {state}: {agent1.get_q_value(state, 1, 0)}')
#print(f'for state {state}: {agent1.get_q_value(state, 1, 1)}')
#print(f'for state {state}: {agent1.get_q_value(state, 1, 2)}')
#print(f'for state {state}: {agent1.get_q_value(state, 1, 3)}')
#print(f'for state {state}: {agent1.get_q_value(state, 1, 4)}')

print()
print()

#winner = control.match()

#print()

#if winner is None:
#  print('Draw!')
#else:
#  print('The winner is ', winner.get_name())