from simulator import Simulator
from DQN import Agent
import numpy as np


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


if __name__ == "__main__":
    mem_size = 100000
    nb_init = 10000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 11
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 300
    nb_episode_steps = 3000
    detection_rate = 0.5  # Percentage of vehicles that can be detected by the algorithm
    gui = False
    alpha = 0.0001
    gamma = 0.9
    epsilon = 1
    epsilon_end = 0.05
    decay_steps = 100000
    batch_size = 32
    target_update_frequency = 3000
    hour_of_the_day = 8
    # Probability for a car to be generated on a particular route at a certain step
    route_probabilities = [1. / 30] * 12
    file_name = "model_50_high.pt"
    stats_file_name = "model_50_high"
    figure_name = "medium_50_high"

    # Initializing the simulator, agent and replay buffer
    agent = Agent(alpha, gamma, epsilon, epsilon_end, decay_steps, batch_size, nb_inputs, nb_actions, mem_size,
                  file_name)
    print("INITIALIZING REPLAY BUFFER")
    initialize_buffer(agent.replayBuffer, nb_init, nb_actions, nb_episode_steps, detection_rate, route_probabilities,
                      hour_of_the_day)
    print("REPLAY BUFFER INITIALIZATION DONE")
    # /!\ Has to be initialized AFTER buffer initialization! initialize_buffer() uses its own simulator
    simulator = Simulator(nb_episodes, nb_episode_steps, detection_rate, route_probabilities, hour_of_the_day, gui)

    # Learning phase
    print("STARTING LEARNING")
    continue_simulation = True
    while continue_simulation:
        action = agent.select_action(simulator.get_state())
        continue_simulation = collect_transition(agent.replayBuffer, simulator, action)
        agent.learning_step()
        if simulator.currNbIterations % target_update_frequency == 0:
            agent.update_target_net()
    print("LEARNING DONE")

    print("SAVING Q-NET")
    agent.save_net()
    print("DONE")
    print("SAVING STATS")
    simulator.save_stats(stats_file_name, figure_name)
    print("DONE")
