import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    episodes = []
    rewards1 = []
    rewards2 = []
    rewards3 = []
    rewards4 = []
    rewards5 = []
    rewards6 = []
    waiting_times1 = []
    waiting_times2 = []
    waiting_times3 = []
    waiting_times4 = []
    waiting_times5 = []
    waiting_times6 = []
    rewards1_dev = []
    rewards2_dev = []
    rewards3_dev = []
    rewards4_dev = []
    rewards5_dev = []
    rewards6_dev = []
    waiting_times1_dev = []
    waiting_times2_dev = []
    waiting_times3_dev = []
    waiting_times4_dev = []
    waiting_times5_dev = []
    waiting_times6_dev = []
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0

    figure_name = "uniform/pedestrian"

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_low_ped_neglected/events.out.tfevents.1587383116.alan-compute-09.174234.0"):
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

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_low_pedestrian/events.out.tfevents.1587382854.alan-compute-06.183413.0"):
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
            "runs/model_100_low_pedestrian2/events.out.tfevents.1587393383.alan-compute-06.13188.0"):
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
            "runs/model_100_low_pedestrian3/events.out.tfevents.1587394580.PC-CYRIL-LINUX.7007.0"):
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
            "runs/model_100_low_pedestrian4/events.out.tfevents.1587404150.alan-compute-02.6141.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards5.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times5.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards5_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times5_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_low_pedestrian5/events.out.tfevents.1587415808.alan-compute-07.159695.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards6.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times6.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards6_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times6_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pedestrian_baseline/events.out.tfevents.1587381972.PC-CYRIL-LINUX.9028.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    plt.figure()
    plt.grid()
    plt.plot(episodes, rewards1, color="limegreen", label="pedestrian neglected")
    plt.errorbar(episodes, rewards1, yerr=rewards1_dev, color="limegreen", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards2, color="steelblue", label="1")
    plt.errorbar(episodes, rewards2, yerr=rewards2_dev, color="steelblue", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards6, color="gray", label="2")
    plt.errorbar(episodes, rewards6, yerr=rewards6_dev, color="gray", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards3, color="gold", label="3")
    plt.errorbar(episodes, rewards3, yerr=rewards3_dev, color="gold", elinewidth=3, alpha=0.4)
    plt.plot(episodes[:29], rewards4, color="black", label="4")
    plt.errorbar(episodes[:29], rewards4, yerr=rewards4_dev, color="black", elinewidth=3, alpha=0.4)
    plt.plot(episodes, rewards5, color="darkorange", label="5")
    plt.errorbar(episodes, rewards5, yerr=rewards5_dev, color="darkorange", elinewidth=3, alpha=0.4)
    plt.axhline(y=baseline_r, color="r", label="Fixed time (10s)")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_r.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(episodes, waiting_times1, color="limegreen", label="pedestrian neglected")
    plt.errorbar(episodes, waiting_times1, yerr=waiting_times1_dev, color="limegreen", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times2, color="steelblue", label="1")
    plt.errorbar(episodes, waiting_times2, yerr=waiting_times2_dev, color="steelblue", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times6, color="gray", label="2")
    plt.errorbar(episodes, waiting_times6, yerr=waiting_times6_dev, color="gray", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times3, color="gold", label="3")
    plt.errorbar(episodes, waiting_times3, yerr=waiting_times3_dev, color="gold", elinewidth=3, alpha=0.4)
    plt.plot(episodes[:29], waiting_times4, color="black", label="4")
    plt.errorbar(episodes[:29], waiting_times4, yerr=waiting_times4_dev, color="black", elinewidth=3, alpha=0.4)
    plt.plot(episodes, waiting_times5, color="darkorange", label="5")
    plt.errorbar(episodes, waiting_times5, yerr=waiting_times5_dev, color="darkorange", elinewidth=3, alpha=0.4)
    plt.axhline(y=baseline_w, color="r", label="Fixed time (10s)")
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_w.png")
    plt.show()
