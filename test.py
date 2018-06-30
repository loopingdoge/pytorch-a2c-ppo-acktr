from functools import reduce
from random import shuffle
from statistics import stdev
from subprocess import check_output

test_set = {
    'SonicAndKnuckles3-Genesis': [
        "AngelIslandZone.Act2",  
        "FlyingBatteryZone.Act2",
        "HydrocityZone.Act1",
        "LavaReefZone.Act1"
    ],
    'SonicTheHedgehog2-Genesis': [
        "CasinoNightZone.Act2",
        "HillTopZone.Act2",
        "MetropolisZone.Act3"
    ],
    'SonicTheHedgehog-Genesis': [
        "GreenHillZone.Act2",
        "ScrapBrainZone.Act1",
        "SpringYardZone.Act1",
        "StarLightZone.Act3"
    ]
}


test_set_tuples = []
results = {}

for game, levels in test_set.items():
    for level in levels:
        print(game, level)
        out = check_output(['python', 'main.py', '--env-name', 'Sonic-v0-training', '--algo', 'acktr', '--num-processes', '16', '--num-steps', '20', '--num-frames', '30000', '--game', game, '--level', level, '--silent', '--testing']).decode("utf-8") 
        lines = out.split('\n')

        [avg, avg_std] = lines[0].split(' ')
        avg = float(avg)
        avg_std = float(avg_std)

        [final_mean, final_std] = lines[1].split(' ')
        final_mean = float(final_mean)
        final_std = float(final_std)

        print(f'Average score: {avg:f} ± {avg_std:f}')
        print(f'Final best score: {final_mean:f} ± {final_std:f}')
        print()
        
        results[level] = {
            'avg': avg,
            'avg_std': avg_std,
            'final_mean': final_mean,
            'final_std': final_std
        }
        

avgs = list(map(lambda x: results[x]['avg'], results))
aggregated_avgs = reduce(lambda x, y: x + y, avgs, 0) / len(avgs)
aggregated_avgs_std = stdev(avgs)

finals = list(map(lambda x: results[x]['final_mean'], results))
aggregated_finals = reduce(lambda x, y: x + y, finals, 0) / len(finals)
aggregated_finals_std = stdev(finals)

print("\n## RESULTS ##\n")
for key, res in results.items():
    print(f"{key}   {res['avg']:f} ± {res['avg_std']:f}     {res['final_mean']:f} ± {res['final_std']:f}")

print(f"Aggregated   {aggregated_avgs:f} ± {aggregated_avgs_std:f}     {aggregated_finals:f} ± {aggregated_finals_std:f}\n")

print("\n## CSV ##\n")

print('State,Score,Final Score')

for key, res in results.items():
    print(f"{key},{res['avg']:f} ± {res['avg_std']:f},{res['final_mean']:f} ± {res['final_std']:f}")

print(f"Aggregated,{aggregated_avgs:f} ± {aggregated_avgs_std:f},{aggregated_finals:f} ± {aggregated_finals_std:f}\n")