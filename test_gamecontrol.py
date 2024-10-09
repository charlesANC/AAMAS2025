from qlperfectrm import QLPerfectRM
from qlperfectrm2 import QLPerfectRM2
from qlnorm import QLNoRM
from gamecontrol import GameControl

from scipy.special import softmax


def print_actions(agent, state):
    print(f'{agent.get_name()} State ', state)
    
    results = [0, 0, 0, 0, 0]
    for i in range(5):
        results[i] = agent.get_q_value(state, state[1], i)
    results = softmax(results)
    
    print(f'{agent.get_name()}: LEFT {agent1.get_q_value(state, 1, 0)}')
    print(f'{agent.get_name()}: RIGHT {agent1.get_q_value(state, 1, 1)}')
    print(f'{agent.get_name()}: UP {agent1.get_q_value(state, 1, 2)}')
    print(f'{agent.get_name()}: DOWN {agent1.get_q_value(state, 1, 3)}')
    print(f'{agent.get_name()}: DIG {agent1.get_q_value(state, 1, 4)}')
    
    

#agent1 = QLIndependentBelief('A', decorrelate=True)
agent1 = QLPerfectRM2('A')
agent2 = QLPerfectRM('B')
control = GameControl(agent1, agent2, max_frames=10e6, log_interval=50000)

control.train(agent1, print_logs=True)

print()
print()

control.train(agent2, print_logs=True)

#=========================

print()
print()


state = [11, 0, [True, True, True, True, False, False], 10]

print_actions(agent1, state)


print()
print()


state = [11, 1, [True, True, True, True, False, False], 10]

print_actions(agent1, state)

print()
print()


state = [11, 0, [True, True, True, True, False, False], 15]

print_actions(agent1, state)


print()
print()


state = [11, 1, [True, True, True, True, False, False], 15]

print_actions(agent1, state)

print()
print()


state = [10, 0, [True, True, True, True, False, False], 11]

print_actions(agent1, state)


print()
print()


state = [10, 1, [True, True, True, True, False, False], 11]

print_actions(agent1, state)


#====================

print()
print()


state = [11, 0, [True, True, True, True, False, False], 10]

print_actions(agent2, state)


print()
print()


state = [11, 1, [True, True, True, True, False, False], 10]

print_actions(agent2, state)

print()
print()


state = [11, 0, [True, True, True, True, False, False], 15]

print_actions(agent2, state)


print()
print()


state = [11, 1, [True, True, True, True, False, False], 15]

print_actions(agent2, state)

print()
print()


state = [10, 0, [True, True, True, True, False, False], 11]

print_actions(agent2, state)


print()
print()


state = [10, 1, [True, True, True, True, False, False], 11]

print_actions(agent2, state)



#===================


#winner = control.match()

#print()

#if winner is None:
#  print('Draw!')
#else:
#  print('The winner is ', winner.get_name())




    
    