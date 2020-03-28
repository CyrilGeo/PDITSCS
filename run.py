from simulator import Simulator
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
def initialize_buffer(rp_buf, nb_samples, nb_act, nb_ep_steps, det_rate, route_prob, hour_day):
    sim = Simulator(None, nb_ep_steps, det_rate, route_prob, hour_day, False)
    for i in range(nb_samples):
        selected_action = select_random_action(nb_act)
        collect_transition(rp_buf, sim, selected_action)
    sim.close_simulation()
    print("END OF SIMULATION")


def test_agent(sim, tb, nb_ep_test, nb_ep_steps, det_rate, route_prob, hour_day):
    sim.close_simulation()
    print("\nENTERING TESTING PHASE")
    test_sim = Simulator(nb_ep_test, nb_ep_steps, det_rate, route_prob, hour_day)
    while test_sim.step(agent.select_action(test_sim.get_state(), True)):
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
    mem_size = 100000
    nb_init = 10000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 11
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 200
    nb_episodes_test = 30
    nb_episodes_between_tests = 5
    nb_episode_steps = 3000
    detection_rate = 0.2  # Percentage of vehicles that can be detected by the algorithm
    gui = False
    alpha = 0.0001
    gamma = 0.9
    policy = "boltzmann"
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
    route_probabilities = [1. / 30] * 3 + [1. / 60] * 3 + [1. / 30] * 3 + [1. / 60] * 3
    gen_name = "model_boltz_hor_30_60_20"
    file_name = gen_name + ".pt"

    print("LEARNING " + gen_name)

    writer = SummaryWriter(log_dir="runs/" + gen_name)

    # Initializing the simulator, agent and replay buffer
    agent = Agent(alpha, gamma, policy, epsilon, epsilon_end, decay_steps_ep, temp, temp_end, decay_steps_temp,
                  batch_size, nb_inputs, nb_actions, mem_size, file_name)
    print("\nINITIALIZING REPLAY BUFFER")
    initialize_buffer(agent.replayBuffer, nb_init, nb_actions, nb_episode_steps, detection_rate, route_probabilities,
                      hour_of_the_day)
    print("REPLAY BUFFER INITIALIZATION DONE\n")
    # /!\ Has to be initialized AFTER buffer initialization! initialize_buffer() uses its own simulator
    simulator = Simulator(nb_episodes, nb_episode_steps, detection_rate, route_probabilities, hour_of_the_day, gui)

    # Learning phase
    print("STARTING LEARNING")
    continue_simulation = True
    while continue_simulation:
        if simulator.get_episode_end() == 1 and (simulator.episodeCnt - 1) % nb_episodes_between_tests == 0:
            test_agent(simulator, writer, nb_episodes_test, nb_episode_steps, detection_rate, route_probabilities,
                       hour_of_the_day)
        action = agent.select_action(simulator.get_state())
        continue_simulation = collect_transition(agent.replayBuffer, simulator, action)
        agent.learning_step()
        if simulator.currNbIterations % target_update_frequency == 0:
            agent.update_target_net()
    test_agent(simulator, writer, nb_episodes_test, nb_episode_steps, detection_rate, route_probabilities,
               hour_of_the_day)
    print("LEARNING DONE")

    writer.close()

    print("SAVING Q-NET")
    agent.save_net()
    print("DONE")
    '''print("SAVING STATS")
    simulator.save_stats(gen_name)
    print("DONE")'''

    simulator.delete_sim_files()

    print("FINISHED LEARNING " + gen_name)
