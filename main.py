import time

from agents import Agents
from agents.arguments import get_args
from copy import deepcopy


def multiple_agents(n_agents):
    args = get_args()

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    agents = [Agents(args) for i in range(n_agents)]

    for a in agents:
        a.init_training()
    
    start = time.time()

    for j in range(num_updates):
        params = list(map(lambda a: dict(a.compute_steps()), agents))

        params_sum = deepcopy(params[0])
        for p in params[1:]:
            for name, param in p.items():
                params_sum[name].data.copy_(params_sum[name].data + param.data)

        params_average = deepcopy(params_sum)
        for name, param in params_sum.items():
            params_average[name].data.copy_(params_sum[name].data * 0.5)

        for a in agents:
            a.substitute_params(params_average)

        if j % args.save_interval == 0 and args.save_dir != "":
            agents[0].save_model()

            if j % args.log_interval == 0:
                agents[0].print_progress(start, j)

            if args.vis and j % args.vis_interval == 0:
                agents[0].plot()


if __name__ == "__main__":
    multiple_agents(3)
