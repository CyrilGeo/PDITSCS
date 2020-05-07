import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    rewards1 = []
    waiting_times1 = []
    rewards1_dev = []
    waiting_times1_dev = []
    rewards2 = []
    waiting_times2 = []
    rewards2_dev = []
    waiting_times2_dev = []
    rewards3 = []
    waiting_times3 = []
    rewards3_dev = []
    waiting_times3_dev = []
    rewards4 = []
    waiting_times4 = []
    rewards4_dev = []
    waiting_times4_dev = []
    reward_single100 = 0
    waiting_time_single100 = 0
    reward_single_dev100 = 0
    waiting_time_single_dev100 = 0
    reward_single70 = 0
    waiting_time_single70 = 0
    reward_single_dev70 = 0
    waiting_time_single_dev70 = 0
    reward_single50 = 0
    waiting_time_single50 = 0
    reward_single_dev50 = 0
    waiting_time_single_dev50 = 0
    reward_single20 = 0
    waiting_time_single20 = 0
    reward_single_dev20 = 0
    waiting_time_single_dev20 = 0
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_manhattan_100_medium/events.out.tfevents.1588661109.alan-compute-09.34025.0"):
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
            "runs/model_manhattan_70_medium/events.out.tfevents.1588762120.alan-compute-07.70799.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times2.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards2_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times2_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_manhattan_50_medium/events.out.tfevents.1588763079.alan-compute-09.78372.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times3.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards3_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times3_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_manhattan_20_medium/events.out.tfevents.1588762419.alan-compute-02.1424.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards4.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times4.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards4_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times4_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/manhattan_medium_100/events.out.tfevents.1588704329.PC-CYRIL-LINUX.23099.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward_single100 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time_single100 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward_single_dev100 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time_single_dev100 = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/manhattan_medium_70/events.out.tfevents.1588839684.PC-CYRIL-LINUX.26927.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward_single70 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time_single70 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward_single_dev70 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time_single_dev70 = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/manhattan_medium_50/events.out.tfevents.1588840313.PC-CYRIL-LINUX.27102.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward_single50 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time_single50 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward_single_dev50 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time_single_dev50 = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/manhattan_medium_20/events.out.tfevents.1588840811.PC-CYRIL-LINUX.27277.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward_single20 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time_single20 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward_single_dev20 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time_single_dev20 = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/manhattan_medium_baseline/events.out.tfevents.1588702789.PC-CYRIL-LINUX.22783.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    bar_width = 0.2
    bars1 = [waiting_time_single20, waiting_times4[-1]]
    bars2 = [waiting_time_single50, waiting_times3[-1]]
    bars3 = [waiting_time_single70, waiting_times2[-1]]
    bars4 = [waiting_time_single100, waiting_times1[-1]]
    err1 = [waiting_time_single_dev20, waiting_times4_dev[-1]]
    err2 = [waiting_time_single_dev50, waiting_times3_dev[-1]]
    err3 = [waiting_time_single_dev70, waiting_times2_dev[-1]]
    err4 = [waiting_time_single_dev100, waiting_times1_dev[-1]]
    r1 = [x + 0.4 for x in np.arange(len(bars1))]
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    w_t_dev = [baseline_w_dev, waiting_time_single_dev100, waiting_times1_dev[-1]]
    plt.figure()
    plt.grid(axis="y")
    plt.bar(0, baseline_w, color="r", width=bar_width, edgecolor="white", label="Fixed time", yerr=baseline_w_dev,
            capsize=4)
    plt.bar(r1, bars1, color="gold", width=bar_width, edgecolor="white", label="20% detection rate", yerr=err1,
            capsize=4)
    plt.bar(r2, bars2, color="darkorange", width=bar_width, edgecolor="white", label="50% detection rate", yerr=err2,
            capsize=4)
    plt.bar(r3, bars3, color="steelblue", width=bar_width, edgecolor="white", label="70% detection rate", yerr=err3,
            capsize=4)
    plt.bar(r4, bars4, color="limegreen", width=bar_width, edgecolor="white", label="100% detection rate", yerr=err4,
            capsize=4)
    plt.xticks([r + bar_width + 0.5 for r in range(len(bars1))], ["Single pre-trained agent", "Trained agents"])
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/networks/manhattan_perf.png")
    plt.show()
