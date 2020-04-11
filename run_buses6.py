from priority_simulator import PrioritySimulator
from DQN import Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import statistics


# Randomly selects an action in the action space
def select_random_action(nb_act):
    return np.random.choice([x for x in range(nb_act)])


# Collects one transition from the simulator and stores it into the replay buffer
def collect_transition(replay_buf, simu, act):
    state = simu.get_state()
    continue_sim = simu.step(act)
    reward = simu.get_reward()
    new_state = simu.get_state()
    done = simu.get_episode_end()
    replay_buf.store(np.array(state), act, reward, np.array(new_state), done)
    return continue_sim


# Fills the replay buffer with nb_samples samples using random actions
def initialize_buffer(rp_buf, nb_samples, nb_act, nb_ep_steps, det_rate, min_phase, route_prob, bus_freq_1, bus_freq_2,
                      bus_freq_3, bus_dev, pr_factor, hour_day, hourly_probs):
    sim = PrioritySimulator(None, nb_ep_steps, det_rate, min_phase, route_prob, bus_freq_1, bus_freq_2, bus_freq_3,
                            bus_dev, pr_factor, hour_day, False, hourly_probs)
    for i in range(nb_samples):
        selected_action = select_random_action(nb_act)
        collect_transition(rp_buf, sim, selected_action)
    sim.close_simulation()
    print("END OF SIMULATION")


def test_agent(sim, tb, nb_ep_test, nb_ep_steps, det_rate, min_phase, route_prob, bus_freq_1, bus_freq_2, bus_freq_3,
               bus_dev, pr_factor, hour_day, hourly_probs, ag):
    sim.close_simulation()
    print("\nENTERING TESTING PHASE")
    test_sim = PrioritySimulator(nb_ep_test, nb_ep_steps, det_rate, min_phase, route_prob, bus_freq_1, bus_freq_2,
                                 bus_freq_3, bus_dev, pr_factor, hour_day, False, hourly_probs)
    while test_sim.step(ag.select_action(test_sim.get_state(), True)):
        pass
    av_r = statistics.mean(test_sim.averageRewards)
    av_w = statistics.mean(test_sim.averageWaitingTimes)
    stddev_r = statistics.stdev(test_sim.averageRewards)
    stddev_w = statistics.stdev(test_sim.averageWaitingTimes)
    tb.add_scalar("Average reward", av_r, sim.episodeCnt)
    tb.add_scalar("Average waiting time", av_w, sim.episodeCnt)
    tb.add_scalar("Reward standard deviation", stddev_r, sim.episodeCnt)
    tb.add_scalar("Waiting time standard deviation", stddev_w, sim.episodeCnt)
    print("TESTING DONE")
    print("Average reward:", av_r)
    print("Average waiting time:", av_w)
    print("Reward standard deviation:", stddev_r)
    print("Waiting time standard deviation:", stddev_w)
    print("LEAVING TESTING PHASE\n")
    sim.init_new_episode()


if __name__ == "__main__":
    h_probs = None
    '''h_probs = [[0.0003 / 3] * 3 + [0.0011 / 3] * 3 + [0.0015 / 3] * 3 + [0.0029 / 3] * 3,
               [0.0005 / 3] * 3 + [0.002 / 3] * 3 + [0.0016 / 3] * 3 + [0.002 / 3] * 3,
               [0.0005 / 3] * 3 + [0.0019 / 3] * 3 + [0.0018 / 3] * 3 + [0.0027 / 3] * 3,
               [0.0003 / 3] * 3 + [0.0024 / 3] * 3 + [0.0016 / 3] * 3 + [0.0023 / 3] * 3,
               [0.0035 / 3] * 3 + [0.0032 / 3] * 3 + [0.004 / 3] * 3 + [0.0061 / 3] * 3,
               [0.0197 / 3] * 3 + [0.012 / 3] * 3 + [0.0094 / 3] * 3 + [0.0186 / 3] * 3,
               [0.05 / 3] * 3 + [0.0296 / 3] * 3 + [0.0229 / 3] * 3 + [0.0481 / 3] * 3,
               [0.0717 / 3] * 3 + [0.0507 / 3] * 3 + [0.0375 / 3] * 3 + [0.0653 / 3] * 3,
               [0.0756 / 3] * 3 + [0.0529 / 3] * 3 + [0.0434 / 3] * 3 + [0.0742 / 3] * 3,
               [0.0696 / 3] * 3 + [0.0393 / 3] * 3 + [0.0331 / 3] * 3 + [0.0603 / 3] * 3,
               [0.0541 / 3] * 3 + [0.025 / 3] * 3 + [0.0212 / 3] * 3 + [0.0406 / 3] * 3,
               [0.0284 / 3] * 3 + [0.0208 / 3] * 3 + [0.0184 / 3] * 3 + [0.0293 / 3] * 3,
               [0.0419 / 3] * 3 + [0.0331 / 3] * 3 + [0.0294 / 3] * 3 + [0.0540 / 3] * 3,
               [0.0569 / 3] * 3 + [0.0373 / 3] * 3 + [0.0327 / 3] * 3 + [0.0522 / 3] * 3,
               [0.0465 / 3] * 3 + [0.0263 / 3] * 3 + [0.0244 / 3] * 3 + [0.0542 / 3] * 3,
               [0.0157 / 3] * 3 + [0.0198 / 3] * 3 + [0.0214 / 3] * 3 + [0.0313 / 3] * 3,
               [0.0132 / 3] * 3 + [0.0426 / 3] * 3 + [0.0374 / 3] * 3 + [0.0517 / 3] * 3,
               [0.0156 / 3] * 3 + [0.0594 / 3] * 3 + [0.0575 / 3] * 3 + [0.0702 / 3] * 3,
               [0.0234 / 3] * 3 + [0.0616 / 3] * 3 + [0.0725 / 3] * 3 + [0.0712 / 3] * 3,
               [0.0163 / 3] * 3 + [0.0556 / 3] * 3 + [0.0572 / 3] * 3 + [0.0668 / 3] * 3,
               [0.0087 / 3] * 3 + [0.0363 / 3] * 3 + [0.0328 / 3] * 3 + [0.0387 / 3] * 3,
               [0.0063 / 3] * 3 + [0.0224 / 3] * 3 + [0.0259 / 3] * 3 + [0.0271 / 3] * 3,
               [0.0044 / 3] * 3 + [0.0183 / 3] * 3 + [0.0165 / 3] * 3 + [0.0274 / 3] * 3,
               [0.0037 / 3] * 3 + [0.0171 / 3] * 3 + [0.0196 / 3] * 3 + [0.0256 / 3] * 3]'''

    mem_size = 100000
    nb_init = 10000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 15
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 200
    nb_episodes_test = 30
    nb_episodes_between_tests = 5
    nb_episode_steps = 3000
    detection_rate = 1.0  # Percentage of vehicles that can be detected by the algorithm
    min_phase_duration = 10
    gui = False
    alpha = 0.0001
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
    route_probabilities = [1. / 60] * 12
    bus_frequency_1 = 600
    bus_frequency_2 = 900
    bus_frequency_3 = 600
    bus_stddev = 90
    priority_factor = 5
    gen_name = "model_100_low_buses_pf5"
    file_name = gen_name + ".pt"
    doTesting = True

    print("LEARNING " + gen_name)

    if doTesting:
        writer = SummaryWriter(log_dir="runs/" + gen_name)

    # Initializing the simulator, agent and replay buffer
    agent = Agent(alpha, gamma, policy, epsilon, epsilon_end, decay_steps_ep, temp, temp_end, decay_steps_temp,
                  batch_size, nb_inputs, nb_actions, mem_size, file_name)
    print("\nINITIALIZING REPLAY BUFFER")
    initialize_buffer(agent.replayBuffer, nb_init, nb_actions, nb_episode_steps, detection_rate, min_phase_duration,
                      route_probabilities, bus_frequency_1, bus_frequency_2, bus_frequency_3, bus_stddev,
                      priority_factor, hour_of_the_day, h_probs)
    print("REPLAY BUFFER INITIALIZATION DONE\n")
    # /!\ Has to be initialized AFTER buffer initialization! initialize_buffer() uses its own simulator
    simulator = PrioritySimulator(nb_episodes, nb_episode_steps, detection_rate, min_phase_duration,
                                  route_probabilities, bus_frequency_1, bus_frequency_2, bus_frequency_3, bus_stddev,
                                  priority_factor, hour_of_the_day, gui, h_probs)

    # Learning phase
    print("STARTING LEARNING")
    continue_simulation = True
    while continue_simulation:
        if doTesting and simulator.get_episode_end() == 1 and (
                simulator.episodeCnt - 1) % nb_episodes_between_tests == 0:
            test_agent(simulator, writer, nb_episodes_test, nb_episode_steps, detection_rate, min_phase_duration,
                       route_probabilities, bus_frequency_1, bus_frequency_2, bus_frequency_3, bus_stddev,
                       priority_factor, hour_of_the_day, h_probs, agent)
        action = agent.select_action(simulator.get_state())
        continue_simulation = collect_transition(agent.replayBuffer, simulator, action)
        agent.learning_step()
        if simulator.currNbIterations % target_update_frequency == 0:
            agent.update_target_net()
    test_agent(simulator, writer, nb_episodes_test, nb_episode_steps, detection_rate, min_phase_duration,
               route_probabilities, bus_frequency_1, bus_frequency_2, bus_frequency_3, bus_stddev, priority_factor,
               hour_of_the_day, h_probs, agent)
    print("LEARNING DONE")

    if doTesting:
        writer.close()

    print("SAVING Q-NET")
    agent.save_net()
    print("DONE")
    '''print("SAVING STATS")
    simulator.save_stats(gen_name)
    print("DONE")'''

    simulator.delete_sim_files()

    print("FINISHED LEARNING " + gen_name)
