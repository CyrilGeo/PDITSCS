import matplotlib.pyplot as plt
import tensorflow as tf


def filter_list(w_t):
    new_list = []
    for i in range(len(w_t)):
        if i % 2 == 0:
            new_list.append(w_t[i])
    return new_list


if __name__ == "__main__":
    waiting_times_100 = []
    waiting_times_70 = []
    waiting_times_50 = []
    waiting_times_20 = []
    waiting_times_100_dev = []
    waiting_times_70_dev = []
    waiting_times_50_dev = []
    waiting_times_20_dev = []
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test0/events.out.tfevents.1588687150.PC-CYRIL-LINUX.15572.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test10/events.out.tfevents.1588687227.PC-CYRIL-LINUX.15605.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test20/events.out.tfevents.1588687327.PC-CYRIL-LINUX.15658.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test30/events.out.tfevents.1588687405.PC-CYRIL-LINUX.15702.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test40/events.out.tfevents.1588687472.PC-CYRIL-LINUX.15731.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test50/events.out.tfevents.1588687559.PC-CYRIL-LINUX.15759.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test60/events.out.tfevents.1588687636.PC-CYRIL-LINUX.15808.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test70/events.out.tfevents.1588687718.PC-CYRIL-LINUX.15840.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test80/events.out.tfevents.1588687808.PC-CYRIL-LINUX.15910.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test90/events.out.tfevents.1588687895.PC-CYRIL-LINUX.16021.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100/events.out.tfevents.1588519846.PC-CYRIL-LINUX.27597.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test0/events.out.tfevents.1588688034.PC-CYRIL-LINUX.16075.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test10/events.out.tfevents.1588688141.PC-CYRIL-LINUX.16144.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test20/events.out.tfevents.1588688235.PC-CYRIL-LINUX.16210.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test30/events.out.tfevents.1588688338.PC-CYRIL-LINUX.16262.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test40/events.out.tfevents.1588688423.PC-CYRIL-LINUX.16317.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test50/events.out.tfevents.1588688533.PC-CYRIL-LINUX.16348.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test60/events.out.tfevents.1588688612.PC-CYRIL-LINUX.16396.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70/events.out.tfevents.1588520829.PC-CYRIL-LINUX.28353.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test80/events.out.tfevents.1588688700.PC-CYRIL-LINUX.16419.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test90/events.out.tfevents.1588688821.PC-CYRIL-LINUX.16468.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_70_test100/events.out.tfevents.1588688936.PC-CYRIL-LINUX.16515.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_70.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_70_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test0/events.out.tfevents.1588689051.PC-CYRIL-LINUX.16560.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test10/events.out.tfevents.1588689153.PC-CYRIL-LINUX.16670.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test20/events.out.tfevents.1588689244.PC-CYRIL-LINUX.16733.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test30/events.out.tfevents.1588689352.PC-CYRIL-LINUX.16784.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test40/events.out.tfevents.1588689459.PC-CYRIL-LINUX.16875.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50/events.out.tfevents.1588520652.PC-CYRIL-LINUX.28236.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test60/events.out.tfevents.1588689560.PC-CYRIL-LINUX.16941.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test70/events.out.tfevents.1588689636.PC-CYRIL-LINUX.17067.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test80/events.out.tfevents.1588689723.PC-CYRIL-LINUX.17119.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test90/events.out.tfevents.1588689802.PC-CYRIL-LINUX.17164.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_50_test100/events.out.tfevents.1588689884.PC-CYRIL-LINUX.17210.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_50_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test0/events.out.tfevents.1588690012.PC-CYRIL-LINUX.17269.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test10/events.out.tfevents.1588690150.PC-CYRIL-LINUX.17308.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20/events.out.tfevents.1588520159.PC-CYRIL-LINUX.27738.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test30/events.out.tfevents.1588690312.PC-CYRIL-LINUX.17394.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test40/events.out.tfevents.1588690387.PC-CYRIL-LINUX.17460.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test50/events.out.tfevents.1588690461.PC-CYRIL-LINUX.17495.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test60/events.out.tfevents.1588690571.PC-CYRIL-LINUX.17525.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test70/events.out.tfevents.1588690656.PC-CYRIL-LINUX.17574.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test80/events.out.tfevents.1588690739.PC-CYRIL-LINUX.17626.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test90/events.out.tfevents.1588690815.PC-CYRIL-LINUX.17667.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_20_test100/events.out.tfevents.1588690888.PC-CYRIL-LINUX.17704.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_20_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_baseline/events.out.tfevents.1587729370.PC-CYRIL-LINUX.11624.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    figure_name = "robustness_det_rate"
    detection_rate = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    waiting_times_100 = filter_list(waiting_times_100)
    waiting_times_100_dev = filter_list(waiting_times_100_dev)
    waiting_times_70 = filter_list(waiting_times_70)
    waiting_times_70_dev = filter_list(waiting_times_70_dev)
    waiting_times_50 = filter_list(waiting_times_50)
    waiting_times_50_dev = filter_list(waiting_times_50_dev)
    waiting_times_20 = filter_list(waiting_times_20)
    waiting_times_20_dev = filter_list(waiting_times_20_dev)

    plt.figure()
    plt.grid()
    plt.plot(detection_rate, waiting_times_100, color="limegreen", marker='o', label="100% detection rate")
    plt.errorbar(detection_rate, waiting_times_100, yerr=waiting_times_100_dev, color="limegreen", capsize=4)
    plt.plot(detection_rate, waiting_times_70, color="steelblue", marker='o', label="70% detection rate")
    plt.errorbar(detection_rate, waiting_times_70, yerr=waiting_times_70_dev, color="steelblue", capsize=4)
    plt.plot(detection_rate, waiting_times_50, color="darkorange", marker='o', label="50% detection rate")
    plt.errorbar(detection_rate, waiting_times_50, yerr=waiting_times_50_dev, color="darkorange", capsize=4)
    plt.plot(detection_rate, waiting_times_20, color="gold", marker='o', label="20% detection rate")
    plt.errorbar(detection_rate, waiting_times_20, yerr=waiting_times_20_dev, color="gold", capsize=4)
    plt.axhline(y=baseline_w, color="r", label="Fixed time (10s)")
    plt.xlabel("Detection rate (%)")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name + ".png")
    plt.show()
