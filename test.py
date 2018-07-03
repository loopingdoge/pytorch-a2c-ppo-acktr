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

num_steps = 1000000
env_name = 'Sonic-v0Multiple'
algo = 'acktr'
recurrent = ''

if algo == 'a2c' or algo == 'ppo':
    recurrent = '--recurrent-policy'

test_set_tuples = []
results = {}

for game, levels in test_set.items():
    for level in levels:
        print(game, level)

        execution = [
            'python',           'main.py',
            '--env-name',       env_name,
            '--algo',           algo,
            '--num-processes',  '16',
            '--num-steps',      '20',
            '--num-frames',     str(num_steps),
            '--game',           game,
            '--level',          level,
            '--vis-interval',   '50',
            '--silent',
            '--testing',
            '--record', './record'
        ]
        if len(recurrent) > 0:
            execution.append(recurrent)

        out = check_output(execution).decode("utf-8") 
        
        lines = out.split('\n')

        [avg, avg_std] = lines[-3].split(' ')
        avg = float(avg) * 100
        avg_std = float(avg_std) * 100

        [final_mean, final_std] = lines[-2].split(' ')
        final_mean = float(final_mean) * 100
        final_std = float(final_std) * 100

        print(f'Average score: {avg:.1f} ± {avg_std:.1f}')
        print(f'Final best score: {final_mean:.1f} ± {final_std:.1f}')
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
    print(f"{key}   {res['avg']:.1f} ± {res['avg_std']:.1f}     {res['final_mean']:.1f} ± {res['final_std']:.1f}")

print(f"Aggregated   {aggregated_avgs:.1f} ± {aggregated_avgs_std:.1f}     {aggregated_finals:.1f} ± {aggregated_finals_std:.1f}\n")

print("\n## CSV ##\n")

print('State,Score,Final Score')

for key, res in results.items():
    print(f"{key},{res['avg']:.1f} ± {res['avg_std']:.1f},{res['final_mean']:.1f} ± {res['final_std']:.1f}")

print(f"Aggregated,{aggregated_avgs:.1f} ± {aggregated_avgs_std:.1f},{aggregated_finals:.1f} ± {aggregated_finals_std:.1f}\n")