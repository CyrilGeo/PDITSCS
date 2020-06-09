"""
Training script of the agents for the 5x1 arterial road.
"""

from arterial_sim import ArterialSimulator
from DQN import Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import statistics


def select_random_action(nb_act):
    """
    Randomly selects an action in the action space.
    :param nb_act: the number of actions in the action space
    :return: the selected action
    """
    return np.random.choice([x for x in range(nb_act)])


def collect_transition(replay_buf, simu, act):
    """
    Collects one transition for each agent from the simulator and stores it into their replay buffers.
    :param replay_buf: the replay buffer
    :param simu: the simulator
    :param act: the action to perform to make the transition
    :return: boolean determining whether the simulation is to be continued or not
    """
    states = simu.get_state()
    continue_sim = simu.step(act)
    rewards = simu.get_reward()
    new_states = simu.get_state()
    done = simu.get_episode_end()
    for i in range(5):
        replay_buf[i].store(np.array(states[i]), act[i], rewards[i], np.array(new_states[i]), done)
    return continue_sim


def initialize_buffer(rp_buf, nb_samples, nb_act, nb_ep_steps, det_rate, min_phase, route_prob, hour_day):
    """
    Fills the replay buffer of each agent with nb_samples transitions using random actions.
    :param rp_buf: the replay buffer
    :param nb_samples: the number of transitions to collect
    :param nb_act: the number of actions in the action space
    :param nb_ep_steps: the number of time steps in an episode
    :param det_rate: the detection rate
    :param min_phase: the minimum time of a traffic light phase
    :param route_prob: the route generation probabilities in the intersection
    :param hour_day: the hour of the day
    :return: None
    """
    sim = ArterialSimulator(None, nb_ep_steps, det_rate, min_phase, route_prob, hour_day, False)
    for i in range(nb_samples):
        selected_actions = [select_random_action(nb_act) for x in range(5)]
        collect_transition(rp_buf, sim, selected_actions)
    sim.close_simulation()
    print("END OF SIMULATION")


def test_agent(sim, tb, nb_ep_test, nb_ep_steps, det_rate, min_phase, route_prob, hour_day, ag):
    """
    Tests the performances of the agents on a determined number of episodes.
    :param sim: the simulator
    :param tb: the summary writer for tensorboard
    :param nb_ep_test: the number of test episodes
    :param nb_ep_steps: the number of time steps in an episode
    :param det_rate: the detection rate
    :param min_phase: the minimum time of a traffic light phase
    :param route_prob: the route generation probabilities in the intersection
    :param hour_day: the hour of the day
    :param ag: the agent
    :return: None
    """
    sim.close_simulation()
    print("\nENTERING TESTING PHASE")
    test_sim = ArterialSimulator(nb_ep_test, nb_ep_steps, det_rate, min_phase, route_prob, hour_day, False)
    while test_sim.step([ag[x].select_action(test_sim.get_state()[x], True) for x in range(5)]):
        pass
    av_r = statistics.mean(test_sim.averageRewards)
    av_w = statistics.mean(test_sim.averageWaitingTimes)
    stddev_r = statistics.stdev(test_sim.averageRewards)
    stddev_w = statistics.stdev(test_sim.averageWaitingTimes)
    av_w_det = statistics.mean(test_sim.averageWaitingTimesDetected)
    av_w_undet = statistics.mean(test_sim.averageWaitingTimesUndetected)
    av_w_det_dev = statistics.stdev(test_sim.averageWaitingTimesDetected)
    av_w_undet_dev = statistics.stdev(test_sim.averageWaitingTimesUndetected)
    tb.add_scalar("Average reward", av_r, sim.episodeCnt)
    tb.add_scalar("Average waiting time", av_w, sim.episodeCnt)
    tb.add_scalar("Reward standard deviation", stddev_r, sim.episodeCnt)
    tb.add_scalar("Waiting time standard deviation", stddev_w, sim.episodeCnt)
    tb.add_scalar("Average waiting time detected", av_w_det, sim.episodeCnt)
    tb.add_scalar("Average waiting time undetected", av_w_undet, sim.episodeCnt)
    tb.add_scalar("Waiting time standard deviation detected", av_w_det_dev, sim.episodeCnt)
    tb.add_scalar("Waiting time standard deviation undetected", av_w_undet_dev, sim.episodeCnt)
    print("TESTING DONE")
    print("Average reward:", av_r)
    print("Average waiting time:", av_w)
    print("Reward standard deviation:", stddev_r)
    print("Waiting time standard deviation:", stddev_w)
    print("LEAVING TESTING PHASE\n")
    sim.init_new_episode()


if __name__ == "__main__":
    mem_size = 100000
    nb_init = 10000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 11
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 200
    nb_episodes_test = 10
    nb_episodes_between_tests = 5
    nb_episode_steps = 3000
    detection_rate = 0.7  # Percentage of vehicles that can be detected by the algorithm
    min_phase_duration = 10
    gui = False
    alpha = 0.0001
    milestones = [50, 100]
    lr_decay_factor = 0.1
    gamma = 0.9
    policy = "epsilon-greedy"
    epsilon = 1
    epsilon_end = 0.05
    decay_steps_ep = 100000
    temp = 1
    temp_end = 0.05
    decay_steps_temp = 100000
    batch_size = 32
    target_update_frequency = 3000
    hour_of_the_day = 8
    # Probability for a car to be generated on a particular route at a certain step
    route_probabilities = [1. / 220] * 132
    gen_name = "model_arterial_70_medium"
    file_names = [gen_name + "_" + str(x) + ".pt" for x in range(5)]
    doTesting = True

    print("LEARNING " + gen_name)

    if doTesting:
        writer = SummaryWriter(log_dir="runs/" + gen_name)

    # Initializing the simulator, agents and replay buffers
    agents = []
    for x in range(5):
        agents.append(
            Agent(alpha, milestones, lr_decay_factor, gamma, policy, epsilon, epsilon_end, decay_steps_ep, temp,
                  temp_end, decay_steps_temp, batch_size, nb_inputs, nb_actions, mem_size, file_names[x]))
    print("\nINITIALIZING REPLAY BUFFERS")
    initialize_buffer([agents[x].replayBuffer for x in range(5)], nb_init, nb_actions, nb_episode_steps, detection_rate,
                      min_phase_duration, route_probabilities, hour_of_the_day)
    print("REPLAY BUFFERS INITIALIZATION DONE\n")
    # /!\ Has to be initialized AFTER buffer initialization! initialize_buffer() uses its own simulator
    simulator = ArterialSimulator(nb_episodes, nb_episode_steps, detection_rate, min_phase_duration,
                                  route_probabilities, hour_of_the_day, gui)

    # Learning phase
    print("STARTING LEARNING")
    continue_simulation = True
    while continue_simulation:
        if doTesting and simulator.get_episode_end() == 1 and (
                simulator.episodeCnt - 1) % nb_episodes_between_tests == 0:
            test_agent(simulator, writer, nb_episodes_test, nb_episode_steps, detection_rate, min_phase_duration,
                       route_probabilities, hour_of_the_day, agents)
        actions = [agents[x].select_action(simulator.get_state()[x]) for x in range(5)]
        continue_simulation = collect_transition([agents[x].replayBuffer for x in range(5)], simulator, actions)
        for x in range(5):
            agents[x].learning_step()
            if simulator.get_episode_end() == 1:
                agents[x].scheduling_step()
            if simulator.currNbIterations % target_update_frequency == 0:
                agents[x].update_target_net()
    test_agent(simulator, writer, nb_episodes_test, nb_episode_steps, detection_rate, min_phase_duration,
               route_probabilities, hour_of_the_day, agents)
    print("LEARNING DONE")

    if doTesting:
        writer.close()

    print("SAVING Q-NETS")
    for x in range(5):
        agents[x].save_net()
    print("DONE")

    simulator.delete_sim_files()

    print("FINISHED LEARNING " + gen_name)
