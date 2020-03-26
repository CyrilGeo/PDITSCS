import simulator as sim
import pickle
import statistics
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    nb_episodes = 200
    simulator = sim.Simulator(30, 3000, 1.0, [1. / 45] * 3 + [1. / 60] * 3 + [1. / 45] * 3 + [1. / 60] * 3, 8, False)
    tb = SummaryWriter(log_dir="runs/hor1over45_ver1over60_baseline")

    while simulator.step():
        pass

    reward = statistics.mean(simulator.averageRewards)
    waiting_time = statistics.mean(simulator.averageWaitingTimes)
    stddev_r = statistics.stdev(simulator.averageRewards)
    stddev_w = statistics.stdev(simulator.averageWaitingTimes)
    tb.add_scalar("Average reward", reward, 1)
    tb.add_scalar("Average waiting time", waiting_time, 1)
    tb.add_scalar("Reward standard deviation", stddev_r, 1)
    tb.add_scalar("Waiting time standard deviation", stddev_w, 1)
    tb.add_scalar("Average reward", reward, nb_episodes)
    tb.add_scalar("Average waiting time", waiting_time, nb_episodes)
    tb.add_scalar("Reward standard deviation", stddev_r, nb_episodes)
    tb.add_scalar("Waiting time standard deviation", stddev_w, nb_episodes)
    print("Average reward:", reward)
    print("Average waiting time:", waiting_time)
    print("Reward standard deviation:", stddev_r)
    print("Waiting time standard deviation:", stddev_w)

    tb.close()

    # with open("data/hor1over45_ver1over60_baseline_r.txt", "wb") as file:
    #     pickle.dump(reward, file)
    # with open("data/hor1over45_ver1over60_baseline_w.txt", "wb") as file:
    #     pickle.dump(waiting_time, file)
