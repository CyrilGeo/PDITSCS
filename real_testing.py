# import lux_sim as sim
import lux_sim as sim
import statistics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    simulator = sim.LuxSim(1, 1.0, 5, False)
    while simulator.step():
        '''print(simulator.get_curr_nb_iterations())
        print(simulator.get_state())
        print(simulator.get_reward())'''

    averageHourlyRewards = [statistics.mean(x) for x in simulator.averageHourlyRewards]
    averageHourlyWaitingTimes = [statistics.mean(x) for x in simulator.averageHourlyWaitingTimes]
    hours = [x + 1 for x in range(len(averageHourlyRewards))]

    tb = SummaryWriter(log_dir="runs/hourly_LuST_baseline")
    for i in range(len(hours)):
        tb.add_scalar("Average hourly reward", averageHourlyRewards[i], hours[i])
        tb.add_scalar("Average hourly waiting time", averageHourlyWaitingTimes[i], hours[i])
    tb.close()

    plt.figure()
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
    plt.show()
