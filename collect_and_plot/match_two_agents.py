from extended_gamecontrol import ExtGameControl
from extended_miningenv import ExtMiningEnv
from no_rl_alg.qlrandom import QLRandom
from no_rl_alg.qlstay import QLStay
import pickle



def load_agent(file_name):
    with open(file_name, 'rb') as my_file:
        return pickle.load(my_file)


#agent1 = load_agent('C:\\temp\\aamas2025\models\\QLIndependentBelief_TM_model.pck')
#agent2 = load_agent('C:\\temp\\aamas2025\models\\QLIndependentBelief2_TM_model.pck')
agent1 = load_agent('C:\\temp\\aamas2025\models\\QLIndependentBelief2_TM_model.pck')
agent2 = QLRandom('R', ExtMiningEnv.game_model)
#agent2 = QLStay('S', ExtMiningEnv.game_model)

wins = {agent1.get_name(): 0, agent2.get_name(): 0}


for i in range(50):
    control = ExtGameControl(agent1, agent2, max_frames=5e6, log_interval=10000)


    print()
    print()

    inverse = (i%2==0)
    winner = control.match(inverse=inverse)

    print()

    if winner is None:
        print('Draw!')
    else:
        print('The winner is ', winner.get_name())
        wins[winner.get_name()] = wins[winner.get_name()] + 1
        
        
print('Games won: ', wins)
        