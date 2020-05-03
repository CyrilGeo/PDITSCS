import matplotlib.pyplot as plt
import tensorflow as tf


def filter_list(waiting_times):
    new_list = []
    for i in range(len(waiting_times)):
        if i % 2 == 0:
            new_list.append(waiting_times[i])
    return new_list


def exponential_moving_average(waiting_times, smoothing):
    new_list = []
    prev_val = 0
    new_val = waiting_times[0]
    for i in range(len(waiting_times)):
        if i != 0:
            new_val = waiting_times[i] * smoothing / (1 + i) + prev_val * (1 - smoothing / (1 + i))
        new_list.append(new_val)
        prev_val = new_val
    return new_list


if __name__ == "__main__":

    waiting_times_100 = []
    waiting_times_cars_100 = []
    waiting_times_buses_100 = []
    waiting_times_70 = []
    waiting_times_cars_70 = []
    waiting_times_buses_70 = []
    waiting_times_50 = []
    waiting_times_cars_50 = []
    waiting_times_buses_50 = []
    waiting_times_20 = []
    waiting_times_cars_20 = []
    waiting_times_buses_20 = []
    waiting_times_100_dev = []
    waiting_times_cars_100_dev = []
    waiting_times_buses_100_dev = []
    waiting_times_70_dev = []
    waiting_times_cars_70_dev = []
    waiting_times_buses_70_dev = []
    waiting_times_50_dev = []
    waiting_times_cars_50_dev = []
    waiting_times_buses_50_dev = []
    waiting_times_20_dev = []
    waiting_times_cars_20_dev = []
    waiting_times_buses_20_dev = []
    waiting_time_baseline = 0

    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    tmp5 = []
    tmp6 = []

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf1_buses_baseline/events.out.tfevents.1588429667.PC-CYRIL-LINUX.4480.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_time_baseline = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf1/events.out.tfevents.1588427018.alan-compute-09.181894.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf2/events.out.tfevents.1588427096.alan-compute-09.182068.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf3/events.out.tfevents.1588427096.alan-compute-09.182181.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf4/events.out.tfevents.1588427189.alan-compute-09.182302.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf5/events.out.tfevents.1588427194.alan-compute-09.182422.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf6/events.out.tfevents.1588427290.alan-compute-09.182577.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf7/events.out.tfevents.1588427301.alan-compute-09.182667.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf8/events.out.tfevents.1588426885.alan-compute-07.102411.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf9/events.out.tfevents.1588426921.alan-compute-07.102440.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf10/events.out.tfevents.1588426965.alan-compute-07.102468.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf11/events.out.tfevents.1588427011.alan-compute-07.102497.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf12/events.out.tfevents.1588427042.alan-compute-07.102527.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf13/events.out.tfevents.1588427082.alan-compute-07.102555.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf14/events.out.tfevents.1588427356.alan-compute-02.12460.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf15/events.out.tfevents.1588427286.alan-compute-03.561.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_100.append(tmp1[-1])
    waiting_times_cars_100.append(tmp2[-1])
    waiting_times_buses_100.append(tmp3[-1])
    waiting_times_100_dev.append(tmp4[-1])
    waiting_times_cars_100_dev.append(tmp5[-1])
    waiting_times_buses_100_dev.append(tmp6[-1])

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf1/events.out.tfevents.1588436207.alan-compute-09.203204.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf2/events.out.tfevents.1588436229.alan-compute-09.203252.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf3/events.out.tfevents.1588436229.alan-compute-09.203284.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf4/events.out.tfevents.1588436229.alan-compute-09.203317.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf5/events.out.tfevents.1588436229.alan-compute-09.203347.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf6/events.out.tfevents.1588436229.alan-compute-09.203384.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf7/events.out.tfevents.1588435814.alan-compute-06.97224.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf8/events.out.tfevents.1588435814.alan-compute-06.97272.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf9/events.out.tfevents.1588435814.alan-compute-06.97315.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf10/events.out.tfevents.1588435827.alan-compute-07.103308.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf11/events.out.tfevents.1588435827.alan-compute-07.103327.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf12/events.out.tfevents.1588435837.alan-compute-07.103362.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf13/events.out.tfevents.1588435845.alan-compute-07.103387.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf14/events.out.tfevents.1588435852.alan-compute-07.103412.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf15/events.out.tfevents.1588435863.alan-compute-07.103438.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_70.append(tmp1[-1])
    waiting_times_cars_70.append(tmp2[-1])
    waiting_times_buses_70.append(tmp3[-1])
    waiting_times_70_dev.append(tmp4[-1])
    waiting_times_cars_70_dev.append(tmp5[-1])
    waiting_times_buses_70_dev.append(tmp6[-1])

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf1/events.out.tfevents.1588436795.alan-compute-02.22475.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf2/events.out.tfevents.1588436821.alan-compute-02.22552.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf3/events.out.tfevents.1588436717.alan-compute-03.8758.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf4/events.out.tfevents.1588436765.alan-compute-03.8989.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf5/events.out.tfevents.1588436776.alan-compute-03.9091.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf6/events.out.tfevents.1588436778.alan-compute-03.9172.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf7/events.out.tfevents.1588440876.alan-compute-07.103912.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf8/events.out.tfevents.1588440915.alan-compute-07.103947.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf9/events.out.tfevents.1588440938.alan-compute-07.103977.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf10/events.out.tfevents.1588440950.alan-compute-07.104011.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf11/events.out.tfevents.1588440956.alan-compute-07.104042.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf12/events.out.tfevents.1588440965.alan-compute-07.104072.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf13/events.out.tfevents.1588441251.alan-compute-06.113082.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf14/events.out.tfevents.1588441334.alan-compute-06.113358.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf15/events.out.tfevents.1588441688.alan-compute-09.214258.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_50.append(tmp1[-1])
    waiting_times_cars_50.append(tmp2[-1])
    waiting_times_buses_50.append(tmp3[-1])
    waiting_times_50_dev.append(tmp4[-1])
    waiting_times_cars_50_dev.append(tmp5[-1])
    waiting_times_buses_50_dev.append(tmp6[-1])

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf1/events.out.tfevents.1588445061.alan-compute-06.124117.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf2/events.out.tfevents.1588445365.alan-compute-09.223345.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf3/events.out.tfevents.1588445365.alan-compute-09.223391.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf4/events.out.tfevents.1588445378.alan-compute-09.223442.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf5/events.out.tfevents.1588445385.alan-compute-09.223491.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf6/events.out.tfevents.1588445398.alan-compute-09.223532.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf7/events.out.tfevents.1588445402.alan-compute-09.223576.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf8/events.out.tfevents.1588445341.alan-compute-02.1110.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf9/events.out.tfevents.1588445341.alan-compute-02.1172.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf10/events.out.tfevents.1588445265.alan-compute-03.30440.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf11/events.out.tfevents.1588445273.alan-compute-03.30659.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf12/events.out.tfevents.1588445294.alan-compute-03.30786.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf13/events.out.tfevents.1588445337.alan-compute-03.30880.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf14/events.out.tfevents.1588445927.alan-compute-07.104555.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf15/events.out.tfevents.1588445937.alan-compute-07.104586.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                tmp1.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                tmp2.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                tmp3.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                tmp4.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_cars":
                tmp5.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation_buses":
                tmp6.append(value.simple_value)
    waiting_times_20.append(tmp1[-1])
    waiting_times_cars_20.append(tmp2[-1])
    waiting_times_buses_20.append(tmp3[-1])
    waiting_times_20_dev.append(tmp4[-1])
    waiting_times_cars_20_dev.append(tmp5[-1])
    waiting_times_buses_20_dev.append(tmp6[-1])

    priority_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    figure_name100 = "medium_buses_100"
    figure_name70 = "medium_buses_70"
    figure_name50 = "medium_buses_50"
    figure_name20 = "medium_buses_20"

    '''waiting_times_100 = filter_list(waiting_times_100)
    waiting_times_70 = filter_list(waiting_times_70)
    waiting_times_50 = filter_list(waiting_times_50)
    waiting_times_20 = filter_list(waiting_times_20)
    waiting_times_cars_100 = filter_list(waiting_times_cars_100)
    waiting_times_cars_70 = filter_list(waiting_times_cars_70)
    waiting_times_cars_50 = filter_list(waiting_times_cars_50)
    waiting_times_cars_20 = filter_list(waiting_times_cars_20)
    waiting_times_buses_100 = filter_list(waiting_times_buses_100)
    waiting_times_buses_70 = filter_list(waiting_times_buses_70)
    waiting_times_buses_50 = filter_list(waiting_times_buses_50)
    waiting_times_buses_20 = filter_list(waiting_times_buses_20)'''

    waiting_times_100_smooth = exponential_moving_average(waiting_times_100, 2)
    waiting_times_70_smooth = exponential_moving_average(waiting_times_70, 2)
    waiting_times_50_smooth = exponential_moving_average(waiting_times_50, 2)
    waiting_times_20_smooth = exponential_moving_average(waiting_times_20, 2)
    waiting_times_cars_100_smooth = exponential_moving_average(waiting_times_cars_100, 2)
    waiting_times_cars_70_smooth = exponential_moving_average(waiting_times_cars_70, 2)
    waiting_times_cars_50_smooth = exponential_moving_average(waiting_times_cars_50, 2)
    waiting_times_cars_20_smooth = exponential_moving_average(waiting_times_cars_20, 2)
    waiting_times_buses_100_smooth = exponential_moving_average(waiting_times_buses_100, 2)
    waiting_times_buses_70_smooth = exponential_moving_average(waiting_times_buses_70, 2)
    waiting_times_buses_50_smooth = exponential_moving_average(waiting_times_buses_50, 2)
    waiting_times_buses_20_smooth = exponential_moving_average(waiting_times_buses_20, 2)

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_100, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_100, yerr=waiting_times_100_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_100, color="steelblue", marker='o', label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_100, yerr=waiting_times_cars_100_dev, color="steelblue",
                 capsize=4)
    plt.plot(priority_factors, waiting_times_buses_100, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_100, yerr=waiting_times_buses_100_dev, color="gold", capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name100 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_70, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_70, yerr=waiting_times_70_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_70, color="steelblue", marker='o', label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_70, yerr=waiting_times_cars_70_dev, color="steelblue", capsize=4)
    plt.plot(priority_factors, waiting_times_buses_70, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_70, yerr=waiting_times_buses_70_dev, color="gold", capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name70 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_50, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_50, yerr=waiting_times_50_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_50, color="steelblue", marker='o', label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_50, yerr=waiting_times_cars_50_dev, color="steelblue", capsize=4)
    plt.plot(priority_factors, waiting_times_buses_50, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_50, yerr=waiting_times_buses_50_dev, color="gold", capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name50 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_20, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_20, yerr=waiting_times_20_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_20, color="steelblue", marker='o', label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_20, yerr=waiting_times_cars_20_dev, color="steelblue", capsize=4)
    plt.plot(priority_factors, waiting_times_buses_20, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_20, yerr=waiting_times_buses_20_dev, color="gold", capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name20 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_100_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_100_smooth, yerr=waiting_times_100_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_100_smooth, color="steelblue", marker='o',
             label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_100_smooth, yerr=waiting_times_cars_100_dev, color="steelblue",
                 capsize=4)
    plt.plot(priority_factors, waiting_times_buses_100_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_100_smooth, yerr=waiting_times_buses_100_dev, color="gold",
                 capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name100 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_70_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_70_smooth, yerr=waiting_times_70_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_70_smooth, color="steelblue", marker='o',
             label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_70_smooth, yerr=waiting_times_cars_70_dev, color="steelblue",
                 capsize=4)
    plt.plot(priority_factors, waiting_times_buses_70_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_70_smooth, yerr=waiting_times_buses_70_dev, color="gold",
                 capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name70 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_50_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_50_smooth, yerr=waiting_times_50_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_50_smooth, color="steelblue", marker='o',
             label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_50_smooth, yerr=waiting_times_cars_50_dev, color="steelblue",
                 capsize=4)
    plt.plot(priority_factors, waiting_times_buses_50_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_50_smooth, yerr=waiting_times_buses_50_dev, color="gold",
                 capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name50 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_20_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_20_smooth, yerr=waiting_times_20_dev, color="limegreen", capsize=4)
    plt.plot(priority_factors, waiting_times_cars_20_smooth, color="steelblue", marker='o',
             label="All vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_20_smooth, yerr=waiting_times_cars_20_dev, color="steelblue",
                 capsize=4)
    plt.plot(priority_factors, waiting_times_buses_20_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_20_smooth, yerr=waiting_times_buses_20_dev, color="gold",
                 capsize=4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name20 + "_smooth.png")
    plt.show()
