from acktr import ACKTR


from arguments import get_args

def main():
    args = get_args()
    acktr = ACKTR(args)
    acktr.train()

if __name__ == "__main__":
    main()
