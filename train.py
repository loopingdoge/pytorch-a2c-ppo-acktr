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


for game, levels in train_set.items():
    for level in levels:
        call(['python', 'main.py', '--env-name', 'Sonic-v0-training', '--algo', 'acktr', '--num-processes', '16', '--num-steps', '20', '--num-frames', '100000', '--game', game, '--level', level])
