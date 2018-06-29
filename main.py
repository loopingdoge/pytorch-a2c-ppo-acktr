from agents import Agents
from agents.arguments import get_args


def main():
    args = get_args()
    acktr = Agents(args)
    (avg, avg_std), (final_mean, final_std) = acktr.train()
    print(f'Average score: {avg:f} ± {avg_std:f}')
    print(f'Final best score: {final_mean:f} ± {final_std:f}')


if __name__ == "__main__":
    main()
