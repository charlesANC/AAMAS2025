# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:09:46 2024

@author: charl
"""

from extended_miningenv import ExtMiningEnv


env = ExtMiningEnv(["B"])


print('Inicial:')
env.show_agents_position()

# Testing legal movements

env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)

print('Dois para a direita')
env.show_agents_position()

env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)


print('Dois para cima')
env.show_agents_position()

env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)

print('Volta para baixo')
env.show_agents_position()

env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)


print('Dois para a esquerda')
env.show_agents_position()

env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)


print('Dois para a esquerda')
env.show_agents_position()

# Testing ilegal movements


env.step("B", ExtMiningEnv.LEFT)

print('Um para a esquerda')
env.show_agents_position()



env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.LEFT)


print('Um monte para a esquerda')
env.show_agents_position()

env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)

print('Cinco para a direta')
env.show_agents_position()

env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)

print('Cinco para a direta')
env.show_agents_position()


env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)
env.step("B", ExtMiningEnv.UP)


print('Monte para cima')
env.show_agents_position()

env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)
env.step("B", ExtMiningEnv.DOWN)

print('Monte para baixo')
env.show_agents_position()

# Testing mining where theres no gold

state = env.step("B", ExtMiningEnv.DOWN)

print(f'Estado da RM: { state[0][1]}')

state = env.step("B", ExtMiningEnv.DIG)

print(f'Novo estado da RM depois de minerar: { state[0][1]}\n\n')

# Testing mining where there is gold

state = env.reset("B")

print(f'Depois do reset: { state }')

state = env.step("B", ExtMiningEnv.DIG)

print(f'Depois de minerar onde não tem ouro: { state[0][1]}')

# Testing moving carrying gold

env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.RIGHT)
env.step("B", ExtMiningEnv.DOWN)

print('Aqui tem ouro')
env.show_agents_position()

state = env.step("B", ExtMiningEnv.DIG)

print(f'Peguei o ouro: { state[0][1]}\n\n')

# Testing mining and going into the deposit

env.step("B", ExtMiningEnv.LEFT)
env.step("B", ExtMiningEnv.UP)

print('Voltei carregando o ouro')
env.show_agents_position()

env.step("B", ExtMiningEnv.LEFT)
state = env.step("B", ExtMiningEnv.UP)

print(f'Entreguei o ouro: { state }')
env.show_agents_position()
