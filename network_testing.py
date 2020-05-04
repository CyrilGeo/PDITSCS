import arterial_sim as sim
import statistics
from torch.utils.tensorboard import SummaryWriter

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

    nb_episodes = 30
    nb_episode_steps = 3000
    detection_rate = 1.0  # Percentage of vehicles that can be detected by the algorithm
    min_phase_duration = 10
    gui = True
    hour_of_the_day = 8
    # Probability for a car to be generated on a particular route at a certain step
    route_probabilities = [1. / 220] * 121

    simulator = sim.ArterialSimulator(nb_episodes, nb_episode_steps, detection_rate, min_phase_duration,
                                      route_probabilities, hour_of_the_day, gui)

    nb_episodes_baseline = 200

    while simulator.step():
        '''print("Reward for step", str(simulator.get_curr_nb_iterations()) + ":", str(simulator.get_reward()))
        print(simulator.get_state())'''

    reward = statistics.mean(simulator.averageRewards)
    waiting_time = statistics.mean(simulator.averageWaitingTimes)
    stddev_r = statistics.stdev(simulator.averageRewards)
    stddev_w = statistics.stdev(simulator.averageWaitingTimes)

    waiting_time_detected = statistics.mean(simulator.averageWaitingTimesDetected)
    waiting_time_undetected = statistics.mean(simulator.averageWaitingTimesUndetected)
    waiting_time_detected_dev = statistics.stdev(simulator.averageWaitingTimesDetected)
    waiting_time_undetected_dev = statistics.stdev(simulator.averageWaitingTimesUndetected)

    tb = SummaryWriter(log_dir="runs/uniform_1over60_pf1_buses_baseline")

    tb.add_scalar("Average reward", reward, 1)
    tb.add_scalar("Average waiting time", waiting_time, 1)
    tb.add_scalar("Reward standard deviation", stddev_r, 1)
    tb.add_scalar("Waiting time standard deviation", stddev_w, 1)
    tb.add_scalar("Average reward", reward, nb_episodes_baseline)
    tb.add_scalar("Average waiting time", waiting_time, nb_episodes_baseline)
    tb.add_scalar("Reward standard deviation", stddev_r, nb_episodes_baseline)
    tb.add_scalar("Waiting time standard deviation", stddev_w, nb_episodes_baseline)

    tb.add_scalar("Average waiting time detected", waiting_time_detected, 1)
    tb.add_scalar("Average waiting time undetected", waiting_time_undetected, 1)
    tb.add_scalar("Waiting time standard deviation detected", waiting_time_detected_dev, 1)
    tb.add_scalar("Waiting time standard deviation undetected", waiting_time_undetected_dev, 1)
    tb.add_scalar("Average waiting time detected", waiting_time_detected, nb_episodes)
    tb.add_scalar("Average waiting time undetected", waiting_time_undetected, nb_episodes)
    tb.add_scalar("Waiting time standard deviation detected", waiting_time_detected_dev, nb_episodes)
    tb.add_scalar("Waiting time standard deviation undetected", waiting_time_undetected_dev, nb_episodes)

    tb.close()

    print("Average reward:", reward)
    print("Average waiting time:", waiting_time)
    print("Reward standard deviation:", stddev_r)
    print("Waiting time standard deviation:", stddev_w)

    print("Average waiting time for detected vehicles:", waiting_time_detected)
    print("Average waiting time for undetected vehicles:", waiting_time_undetected)
