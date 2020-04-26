from lux_training_sim import LuxTrainingSim
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
def initialize_buffer(rp_buf, nb_samples, nb_act, det_rate, min_phase):
    sim = LuxTrainingSim(None, det_rate, min_phase, False)
    for i in range(nb_samples):
        selected_action = select_random_action(nb_act)
        collect_transition(rp_buf, sim, selected_action)
    sim.close_simulation()
    print("END OF SIMULATION")


def test_agent(sim, tb, nb_ep_test, det_rate, min_phase, ag):
    sim.close_simulation()
    print("\nENTERING TESTING PHASE")
    test_sim = LuxTrainingSim(nb_ep_test, det_rate, min_phase, False)
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
    mem_size = 3000000
    nb_init = 300000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 27
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 300
    nb_episodes_test = 10
    nb_episodes_between_tests = 10
    detection_rate = 0.7  # Percentage of vehicles that can be detected by the algorithm
    min_phase_duration = 5
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
    gen_name = "model_70_real"
    file_name = gen_name + ".pt"
    doTesting = True

    print("LEARNING " + gen_name)

    if doTesting:
        writer = SummaryWriter(log_dir="runs/" + gen_name)

    # Initializing the simulator, agent and replay buffer
    agent = Agent(alpha, milestones, lr_decay_factor, gamma, policy, epsilon, epsilon_end, decay_steps_ep, temp,
                  temp_end, decay_steps_temp, batch_size, nb_inputs, nb_actions, mem_size, file_name)
    print("\nINITIALIZING REPLAY BUFFER")
    initialize_buffer(agent.replayBuffer, nb_init, nb_actions, detection_rate, min_phase_duration)
    print("REPLAY BUFFER INITIALIZATION DONE\n")
    # /!\ Has to be initialized AFTER buffer initialization! initialize_buffer() uses its own simulator
    simulator = LuxTrainingSim(nb_episodes, detection_rate, min_phase_duration, gui)

    # Learning phase
    print("STARTING LEARNING")
    continue_simulation = True
    while continue_simulation:
        if doTesting and simulator.get_episode_end() == 1 and (
                simulator.episodeCnt - 1) % nb_episodes_between_tests == 0:
            test_agent(simulator, writer, nb_episodes_test, detection_rate, min_phase_duration, agent)
        action = agent.select_action(simulator.get_state())
        continue_simulation = collect_transition(agent.replayBuffer, simulator, action)
        agent.learning_step()
        if simulator.get_episode_end() == 1:
            agent.scheduling_step()
        if simulator.currNbIterations % target_update_frequency == 0:
            agent.update_target_net()
    test_agent(simulator, writer, nb_episodes_test, detection_rate, min_phase_duration, agent)
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
