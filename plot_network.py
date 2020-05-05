import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    rewards1 = []
    waiting_times1 = []
    rewards1_dev = []
    waiting_times1_dev = []
    reward_single = 0
    waiting_time_single = 0
    reward_single_dev = 0
    waiting_time_single_dev = 0
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_arterial_100_medium/events.out.tfevents.1588660017.alan-compute-05.28960.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards1.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times1.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards1_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times1_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/arterial_medium_100/events.out.tfevents.1588705179.PC-CYRIL-LINUX.23483.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward_single = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time_single = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward_single_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time_single_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/arterial_medium_baseline/events.out.tfevents.1588705927.PC-CYRIL-LINUX.24321.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    x = ["Fixed time", "Single pre-trained agent", "Trained agents"]
    w_t = [baseline_w, waiting_time_single, waiting_times1[-1]]
    w_t_dev = [baseline_w_dev, waiting_time_single_dev, waiting_times1_dev[-1]]
    plt.figure()
    plt.grid(axis="y")
    plt.bar(x, w_t, 0.4, yerr=w_t_dev, color=["r", "steelblue", "limegreen"],
            capsize=7)
    plt.ylabel("Average waiting time (s)")
    plt.savefig("figures/waiting_time/networks/arterial_perf.png")
    plt.show()
