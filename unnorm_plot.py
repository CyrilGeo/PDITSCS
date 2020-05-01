import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    reward1 = 0
    reward2 = 0
    reward3 = 0
    reward4 = 0
    reward5 = 0
    reward6 = 0
    reward7 = 0
    reward8 = 0
    reward9 = 0
    reward10 = 0
    waiting_time1 = 0
    waiting_time2 = 0
    waiting_time3 = 0
    waiting_time4 = 0
    waiting_time5 = 0
    waiting_time6 = 0
    waiting_time7 = 0
    waiting_time8 = 0
    waiting_time9 = 0
    waiting_time10 = 0
    reward1_dev = 0
    reward2_dev = 0
    reward3_dev = 0
    reward4_dev = 0
    reward5_dev = 0
    reward6_dev = 0
    reward7_dev = 0
    reward8_dev = 0
    reward9_dev = 0
    reward10_dev = 0
    waiting_time1_dev = 0
    waiting_time2_dev = 0
    waiting_time3_dev = 0
    waiting_time4_dev = 0
    waiting_time5_dev = 0
    waiting_time6_dev = 0
    waiting_time7_dev = 0
    waiting_time8_dev = 0
    waiting_time9_dev = 0
    waiting_time10_dev = 0

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100/events.out.tfevents.1588348114.PC-CYRIL-LINUX.6298.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward1 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time1 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward1_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time1_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_dist50/events.out.tfevents.1588348189.PC-CYRIL-LINUX.6348.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward2 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time2 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward2_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time2_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_dist200/events.out.tfevents.1588348285.PC-CYRIL-LINUX.6380.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward3 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time3 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward3_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time3_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_phase25/events.out.tfevents.1588348441.PC-CYRIL-LINUX.6440.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward4 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time4 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward4_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time4_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_phase100/events.out.tfevents.1588348575.PC-CYRIL-LINUX.6485.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward5 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time5 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward5_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time5_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_unnormalized/events.out.tfevents.1588347549.PC-CYRIL-LINUX.5990.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward6 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time6 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward6_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time6_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_unnormalized_dist50/events.out.tfevents.1588347641.PC-CYRIL-LINUX.6084.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward7 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time7 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward7_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time7_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_unnormalized_dist200/events.out.tfevents.1588347727.PC-CYRIL-LINUX.6163.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward8 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time8 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward8_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time8_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_unnormalized_phase25/events.out.tfevents.1588347868.PC-CYRIL-LINUX.6199.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward9 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time9 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward9_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time9_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_unnormalized_phase100/events.out.tfevents.1588347972.PC-CYRIL-LINUX.6245.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward10 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time10 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward10_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time10_dev = value.simple_value

    figure_name = "figures/waiting_time/uniform/comp_norm.png"

    bar_width = 0.15
    bars1 = [waiting_time1, waiting_time6]
    bars2 = [waiting_time2, waiting_time7]
    bars3 = [waiting_time3, waiting_time8]
    bars4 = [waiting_time4, waiting_time9]
    bars5 = [waiting_time5, waiting_time10]
    err1 = [waiting_time1_dev, waiting_time6_dev]
    err2 = [waiting_time2_dev, waiting_time7_dev]
    err3 = [waiting_time3_dev, waiting_time8_dev]
    err4 = [waiting_time4_dev, waiting_time9_dev]
    err5 = [waiting_time5_dev, waiting_time10_dev]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    plt.figure()
    plt.grid(axis="y")
    plt.bar(r1, bars1, color="r", width=bar_width, edgecolor="white", label="Original intersection", yerr=err1, capsize=5)
    plt.bar(r2, bars2, color="gold", width=bar_width, edgecolor="white", label="50 meter lanes", yerr=err2, capsize=5)
    plt.bar(r3, bars3, color="darkorange", width=bar_width, edgecolor="white", label="200 meter lanes", yerr=err3, capsize=5)
    plt.bar(r4, bars4, color="steelblue", width=bar_width, edgecolor="white", label="25 second phases", yerr=err4, capsize=5)
    plt.bar(r5, bars5, color="limegreen", width=bar_width, edgecolor="white", label="100 second phases", yerr=err5, capsize=5)
    plt.xticks([r + 2 * bar_width for r in range(len(bars1))], ["Normalized", "Not normalized"])
    plt.ylabel("Waiting time (s)")
    plt.legend()
    plt.savefig(figure_name)
    plt.show()