import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

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
    baseline_r = 0
    baseline_w = 0

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_low/events.out.tfevents.1585155365.PC-CYRIL-LINUX.6284.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                episodes.append(event.step)
                rewards1.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times1.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_low/events.out.tfevents.1585168598.alan-compute-08.217244.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times2.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_low/events.out.tfevents.1585168567.alan-compute-07.215803.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times3.append(value.simple_value)

    # baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_baseline/events.out.tfevents.1585171350.PC-CYRIL-LINUX.11892.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value

    plt.figure()
    plt.plot(episodes, rewards1, color="limegreen", label="100% detection rate")
    plt.plot(episodes, rewards2, color="steelblue", label="50% detection rate")
    plt.plot(episodes, rewards3, color="gold", label="20% detection rate")
    plt.axhline(y=baseline_r, color="r", label="fixed time (10s)")
    plt.ylim(bottom=-0.6)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.legend()
    # plt.savefig("figures/" + figure_name + "_r.png")
    plt.show()

    plt.figure()
    plt.plot(episodes, waiting_times1, color="limegreen", label="100% detection rate")
    plt.plot(episodes, waiting_times2, color="steelblue", label="50% detection rate")
    plt.plot(episodes, waiting_times3, color="gold", label="20% detection rate")
    plt.axhline(y=baseline_w, color="r", label="fixed time (10s)")
    plt.ylim(bottom=1.0, top=8.0)
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    # plt.savefig("figures/" + figure_name + "_w.png")
    plt.show()
