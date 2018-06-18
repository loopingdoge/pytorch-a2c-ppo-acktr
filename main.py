from agents import Agents
from agents.arguments import get_args


def main():
    args = get_args()
    acktr = Agents(args)
    acktr.train()


if __name__ == "__main__":
    main()
