import lux_sim as sim
# import lux_training_sim as sim
import statistics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    simulator = sim.LuxSim(10, 1.0, 5, False)
    nb_episodes = 300
    while simulator.step():
        '''print("iteration:", simulator.get_curr_nb_iterations())
        print(simulator.get_state())
        print(simulator.get_reward())'''

    averageHourlyRewards = [statistics.mean(x) for x in simulator.averageHourlyRewards]
    averageHourlyWaitingTimes = [statistics.mean(x) for x in simulator.averageHourlyWaitingTimes]
    hours = [x + 1 for x in range(len(averageHourlyRewards))]

    reward = statistics.mean(simulator.averageRewards)
    waiting_time = statistics.mean(simulator.averageWaitingTimes)
    stddev_r = statistics.stdev(simulator.averageRewards)
    stddev_w = statistics.stdev(simulator.averageWaitingTimes)

    print("Average reward:", reward)
    print("Average waiting time:", waiting_time)
    print("Reward standard deviation:", stddev_r)
    print("Waiting time standard deviation:", stddev_w)

    tb = SummaryWriter(log_dir="runs/hourly_LuST_baseline")

    tb.add_scalar("Average reward", reward, 1)
    tb.add_scalar("Average waiting time", waiting_time, 1)
    tb.add_scalar("Reward standard deviation", stddev_r, 1)
    tb.add_scalar("Waiting time standard deviation", stddev_w, 1)
    tb.add_scalar("Average reward", reward, nb_episodes)
    tb.add_scalar("Average waiting time", waiting_time, nb_episodes)
    tb.add_scalar("Reward standard deviation", stddev_r, nb_episodes)
    tb.add_scalar("Waiting time standard deviation", stddev_w, nb_episodes)

    for i in range(len(hours)):
        tb.add_scalar("Average hourly reward", averageHourlyRewards[i], hours[i])
        tb.add_scalar("Average hourly waiting time", averageHourlyWaitingTimes[i], hours[i])
    tb.close()

    '''plt.figure()
    plt.grid()
    plt.plot(hours, averageHourlyRewards, color="r", label="fixed time")
    plt.xlabel("Hour of the day")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(hours, averageHourlyWaitingTimes, color="r", label="fixed time")
    plt.xlabel("Hour of the day")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.show()'''
