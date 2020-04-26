import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    hours = []
    rewards1 = []
    rewards2 = []
    rewards3 = []
    rewards4 = []
    waiting_times1 = []
    waiting_times2 = []
    waiting_times3 = []
    waiting_times4 = []

    figure_name = "LuST/hourly_real"

    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_100/events.out.tfevents.1587429917.alan-compute-06.146939.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                hours.append(event.step)
                rewards1.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times1.append(value.simple_value)

    '''for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_training_50/events.out.tfevents.1587420389.PC-CYRIL-LINUX.15876.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times2.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_training_20/events.out.tfevents.1587447880.PC-CYRIL-LINUX.19902.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times3.append(value.simple_value)'''

    for event in tf.compat.v1.train.summary_iterator(
            "runs/hourly_LuST_baseline/events.out.tfevents.1587245670.PC-CYRIL-LINUX.24507.0"):
        for value in event.summary.value:
            if value.tag == "Average_hourly_reward":
                rewards4.append(value.simple_value)
            elif value.tag == "Average_hourly_waiting_time":
                waiting_times4.append(value.simple_value)

    plt.figure()
    plt.grid()
    plt.plot(hours, rewards1, color="limegreen", label="100% detection rate")
    '''plt.plot(hours, rewards2, color="steelblue", label="50% detection rate")
    plt.plot(hours, rewards3, color="gold", label="20% detection rate")'''
    plt.plot(hours, rewards4, color="r", label="Fixed time")
    plt.xlabel("Hour")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_r.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(hours, waiting_times1, color="limegreen", label="100% detection rate")
    '''plt.plot(hours, waiting_times2, color="steelblue", label="50% detection rate")
    plt.plot(hours, waiting_times3, color="gold", label="20% detection rate")'''
    plt.plot(hours, waiting_times4, color="r", label="Fixed time")
    plt.xlabel("Hour")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/" + figure_name + "_w.png")
    plt.show()
