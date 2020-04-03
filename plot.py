import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    '''baseline_name = "hor1over30_ver1over60"
    name1 = "model_boltz_hor_30_60_100"
    name2 = "model_boltz_hor_30_60_50"
    name3 = "model_boltz_hor_30_60_20"
    figure_name = "horizontal_30_60_boltz"

    with open("data/" + name1 + "_episodes.txt", "rb") as file:
        episodes = pickle.load(file)
    with open("data/" + baseline_name + "_baseline_r.txt", "rb") as file:
        baseline_r = pickle.load(file)
    with open("data/" + baseline_name + "_baseline_w.txt", "rb") as file:
        baseline_w = pickle.load(file)
    with open("data/" + name1 + "_rewards.txt", "rb") as file:
        rewards1 = pickle.load(file)
    with open("data/" + name2 + "_rewards.txt", "rb") as file:
        rewards2 = pickle.load(file)
    with open("data/" + name3 + "_rewards.txt", "rb") as file:
        rewards3 = pickle.load(file)
    with open("data/" + name1 + "_waiting_times.txt", "rb") as file:
        waiting_times1 = pickle.load(file)
    with open("data/" + name2 + "_waiting_times.txt", "rb") as file:
        waiting_times2 = pickle.load(file)
    with open("data/" + name3 + "_waiting_times.txt", "rb") as file:
        waiting_times3 = pickle.load(file)'''

    episodes = []
    rewards1 = []
    rewards2 = []
    rewards3 = []
    waiting_times1 = []
    waiting_times2 = []
    waiting_times3 = []
    rewards1_dev = []
    rewards2_dev = []
    rewards3_dev = []
    waiting_times1_dev = []
    waiting_times2_dev = []
    waiting_times3_dev = []
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0
    baseline_adapted_r = 0
    baseline_adapted_w = 0
    baseline_adapted_r_dev = 0
    baseline_adapted_w_dev = 0

    figure_name = "horizontal_45_60"

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_45_60_100/events.out.tfevents.1585228331.alan-compute-07.189313.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                episodes.append(event.step)
                rewards1.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times1.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards1_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times1_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_45_60_50/events.out.tfevents.1585228364.alan-compute-08.44301.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times2.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards2_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times2_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_45_60_20/events.out.tfevents.1585228388.alan-compute-08.44614.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times3.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards3_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times3_dev.append(value.simple_value)

    # baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over45_ver1over60_baseline/events.out.tfevents.1585228968.PC-CYRIL-LINUX.26038.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    # baseline adapted
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over45_ver1over60_adapted/events.out.tfevents.1585918655.PC-CYRIL-LINUX.7236.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_adapted_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_adapted_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_adapted_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_adapted_w_dev = value.simple_value

    plt.figure()
    plt.grid()
    plt.plot(episodes, rewards1, color="limegreen", label="100% detection rate")
    plt.errorbar(episodes, rewards1, yerr=rewards1_dev, color="limegreen", elinewidth=1.5, alpha=0.4)
    plt.plot(episodes, rewards2, color="steelblue", label="50% detection rate")
    plt.errorbar(episodes, rewards2, yerr=rewards2_dev, color="steelblue", elinewidth=1.5, alpha=0.4)
    plt.plot(episodes, rewards3, color="gold", label="20% detection rate")
    plt.errorbar(episodes, rewards3, yerr=rewards3_dev, color="gold", elinewidth=1.5, alpha=0.4)
    plt.axhline(y=baseline_r, color="r", label="fixed time (10s)")
    plt.axhline(y=baseline_adapted_r, color="darkviolet", label="adapted fixed time")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_r.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(episodes, waiting_times1, color="limegreen", label="100% detection rate")
    plt.errorbar(episodes, waiting_times1, yerr=waiting_times1_dev, color="limegreen", elinewidth=1.5, alpha=0.4)
    plt.plot(episodes, waiting_times2, color="steelblue", label="50% detection rate")
    plt.errorbar(episodes, waiting_times2, yerr=waiting_times2_dev, color="steelblue", elinewidth=1.5, alpha=0.4)
    plt.plot(episodes, waiting_times3, color="gold", label="20% detection rate")
    plt.errorbar(episodes, waiting_times3, yerr=waiting_times3_dev, color="gold", elinewidth=1.5, alpha=0.4)
    plt.axhline(y=baseline_w, color="r", label="fixed time (10s)")
    plt.axhline(y=baseline_adapted_w, color="darkviolet", label="adapted fixed time")
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_w.png")
    plt.show()

    '''plt.figure()
    plt.plot(episodes, rewards1_dev, color="limegreen", label="100% detection rate")
    plt.plot(episodes, rewards2_dev, color="steelblue", label="50% detection rate")
    plt.plot(episodes, rewards3_dev, color="gold", label="20% detection rate")
    plt.axhline(y=baseline_r_dev, color="r", label="fixed time (10s)")
    plt.ylim(bottom=0.01, top=0.08)
    plt.xlabel("Episode")
    plt.ylabel("Reward standard deviation")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_r_dev.png")
    plt.show()

    plt.figure()
    plt.plot(episodes, waiting_times1_dev, color="limegreen", label="100% detection rate")
    plt.plot(episodes, waiting_times2_dev, color="steelblue", label="50% detection rate")
    plt.plot(episodes, waiting_times3_dev, color="gold", label="20% detection rate")
    plt.axhline(y=baseline_w_dev, color="r", label="fixed time (10s)")
    plt.ylim(bottom=0.0, top=8.0)
    plt.xlabel("Episode")
    plt.ylabel("Waiting time standard deviation")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_w_dev.png")
    plt.show()'''
