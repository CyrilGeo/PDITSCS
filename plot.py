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
    rewards4 = []
    waiting_times1 = []
    waiting_times2 = []
    waiting_times3 = []
    waiting_times4 = []
    rewards1_dev = []
    rewards2_dev = []
    rewards3_dev = []
    rewards4_dev = []
    waiting_times1_dev = []
    waiting_times2_dev = []
    waiting_times3_dev = []
    waiting_times4_dev = []
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0
    baseline_adapted_r = 0
    baseline_adapted_w = 0
    baseline_adapted_r_dev = 0
    baseline_adapted_w_dev = 0

    figure_location = "uniform/"
    figure_name = "veryhigh"

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_veryhigh/events.out.tfevents.1587752514.alan-compute-08.119841.0"):
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

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_veryhigh/events.out.tfevents.1587754019.alan-compute-09.91306.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times2.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards2_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times2_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_veryhigh/events.out.tfevents.1587753838.alan-compute-05.5294.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times3.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards3_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times3_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_veryhigh/events.out.tfevents.1587755872.alan-compute-06.36963.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards4.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times4.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards4_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times4_dev.append(value.simple_value)

    # baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over15_baseline/events.out.tfevents.1587729529.PC-CYRIL-LINUX.11671.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    # adapted baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_random/events.out.tfevents.1587756799.PC-CYRIL-LINUX.21359.0"):
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
    plt.errorbar(episodes, rewards1, yerr=rewards1_dev, color="limegreen", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards2, color="steelblue", label="70% detection rate")
    plt.errorbar(episodes, rewards2, yerr=rewards2_dev, color="steelblue", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards3, color="darkorange", label="50% detection rate")
    plt.errorbar(episodes, rewards3, yerr=rewards3_dev, color="darkorange", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards4, color="gold", label="20% detection rate")
    plt.errorbar(episodes, rewards4, yerr=rewards4_dev, color="gold", elinewidth=3, alpha=0.4)
    plt.axhline(y=baseline_r, color="r", label="Fixed time (10s)")
    # plt.axhline(y=baseline_adapted_r, color="darkviolet", label="adapted fixed time")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/reward/" + figure_location + figure_name + "_r.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(episodes, waiting_times1, color="limegreen", label="100% detection rate")
    plt.errorbar(episodes, waiting_times1, yerr=waiting_times1_dev, color="limegreen", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times2, color="steelblue", label="70% detection rate")
    plt.errorbar(episodes, waiting_times2, yerr=waiting_times2_dev, color="steelblue", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times3, color="darkorange", label="50% detection rate")
    plt.errorbar(episodes, waiting_times3, yerr=waiting_times3_dev, color="darkorange", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times4, color="gold", label="20% detection rate")
    plt.errorbar(episodes, waiting_times4, yerr=waiting_times4_dev, color="gold", elinewidth=3, alpha=0.4)
    plt.axhline(y=baseline_w, color="r", label="Fixed time (10s)")
    # plt.axhline(y=baseline_adapted_w, color="darkviolet", label="adapted fixed time")
    # plt.ylim(bottom=2, top=7)
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/" + figure_location + figure_name + "_w.png")
    plt.show()

    '''x = ["Random", "Fixed time", "20%", "50%", "70%", "100%"]
    w_t = [baseline_adapted_w, baseline_w, waiting_times4[-1], waiting_times3[-1], waiting_times2[-1], waiting_times1[-1]]
    plt.figure()
    plt.grid()
    plt.bar(x, w_t, 0.8, color=["darkviolet", "r", "gold", "darkorange", "steelblue", "limegreen"])
    plt.ylabel("Average waiting time (s)")
    plt.savefig("figures/waiting_time/" + figure_location + figure_name + ".png")
    plt.show()'''

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
