import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    tmp1 = []
    tmp2 = []
    waiting_times = []
    waiting_times_dev = []
    waiting_time_baseline = 0

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_baseline/events.out.tfevents.1587729370.PC-CYRIL-LINUX.11624.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_time_baseline = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_dist50/events.out.tfevents.1588863125.alan-compute-04.12453.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp2.append(value.simple_value)
    waiting_times.append(tmp1[-1])
    waiting_times_dev.append(tmp2[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_dist75/events.out.tfevents.1588870504.alan-compute-06.214514.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp2.append(value.simple_value)
    waiting_times.append(tmp1[-1])
    waiting_times_dev.append(tmp2[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium/events.out.tfevents.1587741096.alan-compute-09.48426.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp2.append(value.simple_value)
    waiting_times.append(tmp1[-1])
    waiting_times_dev.append(tmp2[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_dist125/events.out.tfevents.1588873780.alan-compute-06.62389.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp2.append(value.simple_value)
    waiting_times.append(tmp1[-1])
    waiting_times_dev.append(tmp2[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_dist150/events.out.tfevents.1588947007.alan-compute-02.31039.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp2.append(value.simple_value)
    waiting_times.append(tmp1[-1])
    waiting_times_dev.append(tmp2[-1])

    length = [50, 75, 100, 125, 150]
    plt.figure()
    plt.grid()
    plt.plot(length, waiting_times, color="limegreen", marker='o', label="100% detection rate")
    plt.errorbar(length, waiting_times, yerr=waiting_times_dev, color="limegreen", capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Length of approaches (m)")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/comp_lane_length.png")
    plt.show()
