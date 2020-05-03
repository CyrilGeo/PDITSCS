import matplotlib.pyplot as plt
import tensorflow as tf


def filter_list(w_t):
    new_list = []
    for i in range(len(w_t)):
        if i % 2 == 0:
            new_list.append(w_t[i])
    return new_list


if __name__ == "__main__":
    waiting_times = []
    waiting_times_det = []
    waiting_times_undet = []
    waiting_times_dev = []
    waiting_times_det_dev = []
    waiting_times_undet_dev = []
    baseline_r = 0
    baseline_w = 0
    baseline_r_dev = 0
    baseline_w_dev = 0

    # Detection rates
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100/events.out.tfevents.1588541061.PC-CYRIL-LINUX.2318.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_90/events.out.tfevents.1588540948.PC-CYRIL-LINUX.2249.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_80/events.out.tfevents.1588540835.PC-CYRIL-LINUX.2194.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_70/events.out.tfevents.1588540729.PC-CYRIL-LINUX.2128.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_60/events.out.tfevents.1588540603.PC-CYRIL-LINUX.2037.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_50/events.out.tfevents.1588540443.PC-CYRIL-LINUX.1928.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_40/events.out.tfevents.1588540318.PC-CYRIL-LINUX.1846.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_30/events.out.tfevents.1588540167.PC-CYRIL-LINUX.1713.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_20/events.out.tfevents.1588539984.PC-CYRIL-LINUX.1611.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_10/events.out.tfevents.1588539875.PC-CYRIL-LINUX.1540.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_0/events.out.tfevents.1588539759.PC-CYRIL-LINUX.1464.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_detected":
                waiting_times_det.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_detected":
                waiting_times_det_dev.append(value.simple_value)
            elif value.tag == "Average_waiting_time_undetected":
                waiting_times_undet.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_undetected":
                waiting_times_undet_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_baseline/events.out.tfevents.1587729414.PC-CYRIL-LINUX.11651.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev = value.simple_value

    figure_name = "comp_det_rate_high"
    detection_rate = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    detection_rate_det = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    detection_rate_undet = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    waiting_times = filter_list(waiting_times)
    waiting_times_dev = filter_list(waiting_times_dev)
    waiting_times_det = filter_list(waiting_times_det)
    waiting_times_det_dev = filter_list(waiting_times_det_dev)
    waiting_times_undet = filter_list(waiting_times_undet)
    waiting_times_undet_dev = filter_list(waiting_times_undet_dev)
    waiting_times.reverse()
    waiting_times_dev.reverse()
    waiting_times_det.reverse()
    waiting_times_det_dev.reverse()
    waiting_times_undet.reverse()
    waiting_times_undet_dev.reverse()
    del waiting_times_det[0]
    del waiting_times_det_dev[0]
    del waiting_times_undet[-1]
    del waiting_times_undet_dev[-1]

    plt.figure()
    plt.grid()
    plt.plot(detection_rate, waiting_times, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(detection_rate, waiting_times, yerr=waiting_times_dev, color="limegreen", capsize=4)
    plt.plot(detection_rate_det, waiting_times_det, color="steelblue", marker='o', label="Detected vehicles")
    plt.errorbar(detection_rate_det, waiting_times_det, yerr=waiting_times_det_dev, color="steelblue",
                 capsize=4)
    plt.plot(detection_rate_undet, waiting_times_undet, color="gold", marker='o', label="Undetected vehicles")
    plt.errorbar(detection_rate_undet, waiting_times_undet, yerr=waiting_times_undet_dev, color="gold", capsize=4)
    plt.axhline(y=baseline_w, color="r", label="Fixed time (10s)")
    plt.xlabel("Detection rate (%)")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name + ".png")
    plt.show()
