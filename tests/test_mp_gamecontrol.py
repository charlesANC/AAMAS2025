from qlperfectrm import QLPerfectRM
from qlperfectrm2 import QLPerfectRM2
from qlnorm import QLNoRM
from qlindependentbelief import QLIndependentBelief
from qlbeliefthresholding import QLBeliefThresholding
from mp_gamecontrol import MultiPlayersGameControl


wins = {'A': 0, 'B': 0}



for i in range(20):
    agent1 = QLPerfectRM('A') if i % 2 == 0 else QLPerfectRM2('B')
    agent2 = QLPerfectRM2('B') if i % 2 == 0 else QLPerfectRM('A')
    #agent2 = QLNoRM('A')
    #agent1 = QLBeliefThresholding('A', movement_cost=0.02)
    #agent1 = QLIndependentBelief('A', decorrelate=False)
    #agent2 = QLIndependentBelief('B', decorrelate=True)

    control = MultiPlayersGameControl([agent1, agent2], max_frames=5e6, log_interval=10000)
    #control = MultiPlayersGameControl([agent2], max_frames=10e6, log_interval=10000)

    control.train_agents(print_logs=True)


    print()
    print()

    winner = control.match()

    print()

    if winner is None:
        print('Draw!')
    else:
        print('The winner is ', winner.get_name())
        wins[winner.get_name()] = wins[winner.get_name()] + 1
        
        
print('Games won: ', wins)
        