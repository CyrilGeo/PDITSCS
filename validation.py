from simulator import Simulator
from DQN import Agent

if __name__ == "__main__":
    mem_size = 100000
    nb_init = 10000  # Number of samples in the replay buffer before learning starts
    nb_inputs = 11
    nb_actions = 2  # Either stay at current phase or switch to the next one
    nb_episodes = 1
    nb_episode_steps = 3000
    detection_rate = 1.0  # Percentage of vehicles that can be detected by the algorithm
    gui = True
    alpha = 0.0001
    gamma = 0.9
    policy = "epsilon-greedy"
    epsilon = 0  # /!\ This value cannot change, validation involves exploitation only
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
    file_name = "model_100_medium.pt"

    simulator = Simulator(nb_episodes, nb_episode_steps, detection_rate, route_probabilities, hour_of_the_day, gui)
    agent = Agent(alpha, gamma, policy, epsilon, epsilon_end, decay_steps_ep, temp, temp_end, decay_steps_temp,
                  batch_size, nb_inputs, nb_actions, mem_size, file_name)
    agent.load_net()
    while simulator.step(agent.select_action(simulator.get_state())):
        print("Reward for step", str(simulator.get_curr_nb_iterations()) + ":", str(simulator.get_reward()))
