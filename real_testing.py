import lux_sim as sim
from DQN import Agent
import statistics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    mem_size = 3000000
    nb_init = 300000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 27
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 5
    nb_episodes_test = 10
    nb_episodes_between_tests = 10
    detection_rate = 0.7  # Percentage of vehicles that can be detected by the algorithm
    min_phase_duration = 5
    burst_frequency = 80
    burst_deviation = 5
    burst_stddev = 15
    burst = True
    gui = False
    alpha = 0.0001
    milestones = [50, 100]
    lr_decay_factor = 0.1
    gamma = 0.9
    policy = "epsilon-greedy"
    epsilon = 1
    epsilon_end = 0.05
    decay_steps_ep = 3000000
    temp = 1
    temp_end = 0.05
    decay_steps_temp = 100000
    batch_size = 32
    target_update_frequency = 3000
    file_name = "model_70_real_burst.pt"

    agent = Agent(alpha, milestones, lr_decay_factor, gamma, policy, epsilon, epsilon_end, decay_steps_ep, temp,
                  temp_end, decay_steps_temp, batch_size, nb_inputs, nb_actions, mem_size, file_name)
    '''simulator = sim.LuxTrainingSim(nb_episodes, detection_rate, min_phase_duration, burst_frequency, burst_deviation,
                                   burst_stddev, burst, gui)'''
    simulator = sim.LuxSim(nb_episodes, detection_rate, min_phase_duration, gui)
    nb_episodes_baseline = 300
    agent.load_net()
    while simulator.step(agent.select_action(simulator.get_state(), True)):
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

    tb = SummaryWriter(log_dir="runs/hourly_LuST_70_burst")

    tb.add_scalar("Average reward", reward, 1)
    tb.add_scalar("Average waiting time", waiting_time, 1)
    tb.add_scalar("Reward standard deviation", stddev_r, 1)
    tb.add_scalar("Waiting time standard deviation", stddev_w, 1)
    tb.add_scalar("Average reward", reward, nb_episodes_baseline)
    tb.add_scalar("Average waiting time", waiting_time, nb_episodes_baseline)
    tb.add_scalar("Reward standard deviation", stddev_r, nb_episodes_baseline)
    tb.add_scalar("Waiting time standard deviation", stddev_w, nb_episodes_baseline)

    for i in range(len(hours)):
        tb.add_scalar("Average hourly reward", averageHourlyRewards[i], hours[i])
        tb.add_scalar("Average hourly waiting time", averageHourlyWaitingTimes[i], hours[i])
    tb.close()

    '''plt.figure()
    plt.grid()
    plt.plot(hours, averageHourlyRewards, color="r", label="Fixed time")
    plt.xlabel("Hour of the day")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(hours, averageHourlyWaitingTimes, color="r", label="Fixed time")
    plt.xlabel("Hour of the day")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.show()'''
