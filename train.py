from random import shuffle
from subprocess import call

train_set = {
    'SonicAndKnuckles3-Genesis': [
        "AngelIslandZone.Act1",
        "FlyingBatteryZone.Act1",
        "HydrocityZone.Act2",
        "LavaReefZone.Act2"
    ],
    'SonicTheHedgehog2-Genesis': [
        "CasinoNightZone.Act1",
        "HillTopZone.Act1",
        "MetropolisZone.Act1",
        "MetropolisZone.Act2"
    ],
    'SonicTheHedgehog-Genesis': [
        "GreenHillZone.Act1",
        "GreenHillZone.Act3",
        "ScrapBrainZone.Act2",
        "SpringYardZone.Act2",
        "SpringYardZone.Act3",
        "StarLightZone.Act1",
        "StarLightZone.Act2"
    ]
}

# train_set = {
#     'SonicTheHedgehog-Genesis': [
#         "GreenHillZone.Act3",
#     ]
# }

num_iters = 1
num_steps = 1000000
env_name = 'Sonic-v0GreenHill'
algo = 'acktr'
recurrent = ''

if algo == 'a2c' or algo == 'ppo':
    recurrent = '--recurrent-policy'

train_set_tuples = []

for game, levels in train_set.items():
    for level in levels:
        train_set_tuples.append((game, level))

for i in range(num_iters):
    shuffle(train_set_tuples)
    for (game, level) in train_set_tuples:
        print(i, game, level)
        
        execution = [
            'python',           'main.py',
            '--env-name',       env_name,
            '--algo',           algo,
            '--num-processes',  '16',
            '--num-steps',      '20',
            '--num-frames',     str(num_steps),
            '--game',           game,
            '--level',          level,
            '--vis-interval',   '50'
        ]
        if len(recurrent) > 0:
            execution.append(recurrent)

        call(execution)