import time

from agents import Agents
from agents.arguments import get_args
from copy import deepcopy

train_set = [("SonicTheHedgehog-Genesis", "SpringYardZone.Act3"),
             ("SonicTheHedgehog-Genesis", "SpringYardZone.Act2"),
             ("SonicTheHedgehog-Genesis", "GreenHillZone.Act3"),
             ("SonicTheHedgehog-Genesis", "GreenHillZone.Act1"),
             ("SonicTheHedgehog-Genesis", "StarLightZone.Act2"),
             ("SonicTheHedgehog-Genesis", "StarLightZone.Act1"),
             ("SonicTheHedgehog-Genesis", "MarbleZone.Act2"),
             ("SonicTheHedgehog-Genesis", "MarbleZone.Act1"),
             ("SonicTheHedgehog-Genesis", "MarbleZone.Act3"),
             ("SonicTheHedgehog-Genesis", "ScrapBrainZone.Act2"),
             ("SonicTheHedgehog-Genesis", "LabyrinthZone.Act2"),
             ("SonicTheHedgehog-Genesis", "LabyrinthZone.Act1"),
             ("SonicTheHedgehog-Genesis", "LabyrinthZone.Act3"),
             ("SonicTheHedgehog2-Genesis", "EmeraldHillZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "EmeraldHillZone.Act2"),
             ("SonicTheHedgehog2-Genesis", "ChemicalPlantZone.Act2"),
             ("SonicTheHedgehog2-Genesis", "ChemicalPlantZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "MetropolisZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "MetropolisZone.Act2"),
             ("SonicTheHedgehog2-Genesis", "OilOceanZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "OilOceanZone.Act2"),
             ("SonicTheHedgehog2-Genesis", "MysticCaveZone.Act2"),
             ("SonicTheHedgehog2-Genesis", "MysticCaveZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "HillTopZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "CasinoNightZone.Act1"),
             ("SonicTheHedgehog2-Genesis", "WingFortressZone"),
             ("SonicTheHedgehog2-Genesis", "AquaticRuinZone.Act2"),
             ("SonicTheHedgehog2-Genesis", "AquaticRuinZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "LavaReefZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "CarnivalNightZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "CarnivalNightZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "MarbleGardenZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "MarbleGardenZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "MushroomHillZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "MushroomHillZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "DeathEggZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "DeathEggZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "FlyingBatteryZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "SandopolisZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "SandopolisZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "HiddenPalaceZone"),
             ("SonicAndKnuckles3-Genesis", "HydrocityZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "IcecapZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "IcecapZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "AngelIslandZone.Act1"),
             ("SonicAndKnuckles3-Genesis", "LaunchBaseZone.Act2"),
             ("SonicAndKnuckles3-Genesis", "LaunchBaseZone.Act1")]


def multiple_agents(train_set):
    args = get_args()

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    train_subset = train_set[:3]

    agents = []
    for i in range(len(train_subset)):
        game, level = train_subset[i]
        agents.append(Agents(args, game, level))

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
            params_average[name].data.copy_(
                params_sum[name].data * (1/len(train_subset)))

        for a in agents:
            a.substitute_params(params_average)

        if j % args.save_interval == 0 and args.save_dir != "":
            agents[0].save_model()

            if j % args.log_interval == 0:
                agents[0].print_progress(start, j)

            if args.vis and j % args.vis_interval == 0:
                agents[0].plot()


if __name__ == "__main__":
    multiple_agents(train_set)
