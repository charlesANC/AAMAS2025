#from qlperfectrm import QLPerfectRM
#from qlperfectrm2 import QLPerfectRM2
#from qlnorm import QLNoRM
#from qlindependentbelief import QLIndependentBelief
#from qlbeliefthresholding import QLBeliefThresholding
from mp_gamecontrol import MultiPlayersGameControl
import pickle


wins = {'A': 0, 'B': 0}

def load_data(file_name):
    with open(file_name, 'rb') as my_file:
        return pickle.load(my_file)

agent2 = load_data('C:\\temp\\aamas2025\\models\\QLNorm_model.pck')
agent1 = load_data('C:\\temp\\aamas2025\\models\\QLIndependentBelief_T_model.pck')

control = MultiPlayersGameControl([agent1, agent2], max_frames=5e6, log_interval=10000)
print()
print()

for i in range(1):


    winner = control.match()

    print()

    if winner is None:
        print('Draw!')
    else:
        print('The winner is ', winner.get_name())
        wins[winner.get_name()] = wins[winner.get_name()] + 1
        
        
print('Games won: ', wins)
        