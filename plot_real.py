import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    hours = []
    rewards1 = []
    rewards2 = []
    rewards3 = []
    rewards4 = []
    rewards5 = []
    rewards1_dev = []
    rewards2_dev = []
    rewards3_dev = []
    rewards4_dev = []
    rewards5_dev = []
    waiting_times1 = []
    waiting_times2 = []
    waiting_times3 = []
    waiting_times4 = []
    waiting_times5 = []
    waiting_times1_dev = []
    waiting_times2_dev = []
    waiting_times3_dev = []
    waiting_times4_dev = []
    waiting_times5_dev = []

    figure_name = "LuST/hourly_real_burst"

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_100_burst/events.out.tfevents.1588520998.alan-compute-01.30095.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                hours.append(event.step)
                rewards1.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times1.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_70_burst/events.out.tfevents.1588733492.alan-compute-02.23330.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times2.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_50_burst/events.out.tfevents.1588733479.alan-compute-02.23802.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times3.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_20_burst/events.out.tfevents.1588750573.alan-compute-03.14020.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards4.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times4.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_baseline/events.out.tfevents.1587245670.PC-CYRIL-LINUX.24507.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards5.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times5.append(value.simple_value)

    plt.figure()
    plt.grid()
    plt.plot(hours, rewards1, color="limegreen", label="100% detection rate")
    plt.plot(hours, rewards2, color="steelblue", label="70% detection rate")
    plt.plot(hours, rewards3, color="darkorange", label="50% detection rate")
    plt.plot(hours, rewards4, color="gold", label="20% detection rate")
    plt.plot(hours, rewards5, color="r", label="Fixed time")
    plt.xlabel("Hour")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/reward/" + figure_name + "_r.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(hours, waiting_times1, color="limegreen", label="100% detection rate")
    plt.plot(hours, waiting_times2, color="steelblue", label="70% detection rate")
    plt.plot(hours, waiting_times3, color="darkorange", label="50% detection rate")
    plt.plot(hours, waiting_times4, color="gold", label="20% detection rate")
    plt.plot(hours, waiting_times5, color="r", label="Fixed time")
    plt.xlabel("Hour")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/" + figure_name + "_w.png")
    plt.show()
