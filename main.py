from agents import Agents
from agents.arguments import get_args

from functools import reduce
from statistics import stdev

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

def main():
    args = get_args()
    
    if args.full_set == False:
        acktr = Agents(args)
        (avg, avg_std), (final_mean, final_std) = acktr.train()
        if args.testing:
            print(f'{avg:f} {avg_std:f}')
            print(f'{final_mean:f} {final_std:f}')
        else:
            print(f'Average score: {avg:f} ± {avg_std:f}')
            print(f'Final best score: {final_mean:f} ± {final_std:f}')
    else:
        results = {}
        for game, levels in test_set.items():
            for level in levels:
                args.game = game
                args.level = level
                acktr = Agents(args)
                (avg, avg_std), (final_mean, final_std) = acktr.train()
                results[level] = {
                    'avg': avg,
                    'avg_std': avg_std,
                    'final_mean': final_mean,
                    'final_std': final_std
                }
                print(f'Average score: {avg:f} ± {avg_std:f}')
                print(f'Final best score: {final_mean:f} ± {final_std:f}')
                acktr = None

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

if __name__ == "__main__":
    main()
