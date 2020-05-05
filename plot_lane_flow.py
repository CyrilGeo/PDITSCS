import matplotlib.pyplot as plt
import tensorflow as tf


def filter_list(w_t):
    new_list = []
    for i in range(len(w_t)):
        if i % 2 == 0:
            new_list.append(w_t[i])
    return new_list


if __name__ == "__main__":
    waiting_times_low = []
    waiting_times_low_dev = []
    waiting_times_medium = []
    waiting_times_medium_dev = []
    waiting_times_high = []
    waiting_times_high_dev = []
    baseline_w = []

    # Low traffic flow
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100/events.out.tfevents.1588539472.PC-CYRIL-LINUX.1373.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test1over150/events.out.tfevents.1588692832.PC-CYRIL-LINUX.18173.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test1over100/events.out.tfevents.1588692904.PC-CYRIL-LINUX.18216.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test1over75/events.out.tfevents.1588693028.PC-CYRIL-LINUX.18243.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test1over60/events.out.tfevents.1588693161.PC-CYRIL-LINUX.18340.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test1over50/events.out.tfevents.1588693275.PC-CYRIL-LINUX.18434.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test7over300/events.out.tfevents.1588693385.PC-CYRIL-LINUX.18501.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test2over75/events.out.tfevents.1588693513.PC-CYRIL-LINUX.18549.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test3over100/events.out.tfevents.1588693706.PC-CYRIL-LINUX.18635.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_100_test1over30/events.out.tfevents.1588693847.PC-CYRIL-LINUX.18683.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_low.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_low_dev.append(value.simple_value)

    # Medium traffic flow
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test1over300/events.out.tfevents.1588694721.PC-CYRIL-LINUX.19079.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test1over150/events.out.tfevents.1588694820.PC-CYRIL-LINUX.19151.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test1over100/events.out.tfevents.1588694910.PC-CYRIL-LINUX.19200.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test1over75/events.out.tfevents.1588694980.PC-CYRIL-LINUX.19247.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100/events.out.tfevents.1588519846.PC-CYRIL-LINUX.27597.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test1over50/events.out.tfevents.1588695120.PC-CYRIL-LINUX.19350.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test7over300/events.out.tfevents.1588695194.PC-CYRIL-LINUX.19399.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test2over75/events.out.tfevents.1588695297.PC-CYRIL-LINUX.19434.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test3over100/events.out.tfevents.1588695391.PC-CYRIL-LINUX.19486.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_test1over30/events.out.tfevents.1588695482.PC-CYRIL-LINUX.19539.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_medium.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_medium_dev.append(value.simple_value)

    # High traffic flow
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test1over300/events.out.tfevents.1588696557.PC-CYRIL-LINUX.19868.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test1over150/events.out.tfevents.1588696741.PC-CYRIL-LINUX.19929.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test1over100/events.out.tfevents.1588696821.PC-CYRIL-LINUX.20021.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test1over75/events.out.tfevents.1588696929.PC-CYRIL-LINUX.20074.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test1over60/events.out.tfevents.1588696999.PC-CYRIL-LINUX.20101.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test1over50/events.out.tfevents.1588697126.PC-CYRIL-LINUX.20149.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test7over300/events.out.tfevents.1588697244.PC-CYRIL-LINUX.20193.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test2over75/events.out.tfevents.1588697371.PC-CYRIL-LINUX.20260.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100_test3over100/events.out.tfevents.1588697536.PC-CYRIL-LINUX.20365.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_100/events.out.tfevents.1588541061.PC-CYRIL-LINUX.2318.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_high.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_high_dev.append(value.simple_value)

    # Baselines
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_baseline/events.out.tfevents.1587729335.PC-CYRIL-LINUX.11597.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over150_baseline/events.out.tfevents.1588697761.PC-CYRIL-LINUX.20461.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over100_baseline/events.out.tfevents.1588697810.PC-CYRIL-LINUX.20503.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over75_baseline/events.out.tfevents.1588697830.PC-CYRIL-LINUX.20521.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_baseline/events.out.tfevents.1587729370.PC-CYRIL-LINUX.11624.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over50_baseline/events.out.tfevents.1588697854.PC-CYRIL-LINUX.20538.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_7over300_baseline/events.out.tfevents.1588697880.PC-CYRIL-LINUX.20581.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_2over75_baseline/events.out.tfevents.1588697907.PC-CYRIL-LINUX.20601.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_3over100_baseline/events.out.tfevents.1588697937.PC-CYRIL-LINUX.20626.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_baseline/events.out.tfevents.1587729414.PC-CYRIL-LINUX.11651.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                baseline_w.append(value.simple_value)

    figure_name = "robustness_traffic_flow"
    traffic_flow = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    waiting_times_low = filter_list(waiting_times_low)
    waiting_times_low_dev = filter_list(waiting_times_low_dev)
    waiting_times_medium = filter_list(waiting_times_medium)
    waiting_times_medium_dev = filter_list(waiting_times_medium_dev)
    waiting_times_high = filter_list(waiting_times_high)
    waiting_times_high_dev = filter_list(waiting_times_high_dev)
    baseline_w = filter_list(baseline_w)

    plt.figure()
    plt.grid()
    plt.plot(traffic_flow, waiting_times_high, color="limegreen", marker='o',
             label="Q-Network trained on a 0.1veh/s per lane flow")
    plt.errorbar(traffic_flow, waiting_times_high, yerr=waiting_times_high_dev, color="limegreen", capsize=4)
    plt.plot(traffic_flow, waiting_times_medium, color="steelblue", marker='o',
             label="Q-Network trained on a 0.05veh/s per lane flow")
    plt.errorbar(traffic_flow, waiting_times_medium, yerr=waiting_times_medium_dev, color="steelblue", capsize=4)
    plt.plot(traffic_flow, waiting_times_low, color="gold", marker='o',
             label="Q-Network trained on a 0.01veh/s per lane flow")
    plt.errorbar(traffic_flow, waiting_times_low, yerr=waiting_times_low_dev, color="gold", capsize=4)
    plt.plot(traffic_flow, baseline_w, color="r", label="Fixed time (10s)")
    plt.xlabel("Traffic flow per lane (vehicles/s)")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name + ".png")
    plt.show()
