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

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf1_buses_baseline/events.out.tfevents.1586625892.PC-CYRIL-LINUX.16867.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_time_baseline = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf1/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf2/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf3/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf4/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf5/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf6/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf7/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf8/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf9/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf10/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf11/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf12/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf13/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf14/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium_buses_pf15/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf1/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf2/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf3/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf4/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf5/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf6/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf7/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf8/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf9/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf10/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf11/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf12/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf13/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf14/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium_buses_pf15/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf1/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf2/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf3/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf4/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf5/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf6/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf7/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf8/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf9/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf10/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf11/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf12/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf13/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf14/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium_buses_pf15/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf1/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf2/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf3/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf4/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf5/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf6/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf7/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf8/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf9/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf10/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf11/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf12/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf13/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf14/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium_buses_pf15/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value[-1])
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_cars":
                waiting_times_cars_100_dev.append(value.simple_value[-1])
            elif value.tag == "Waiting_time_standard_deviation_buses":
                waiting_times_buses_100_dev.append(value.simple_value[-1])

    priority_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    figure_name100 = "medium_buses_100"
    figure_name70 = "medium_buses_70"
    figure_name50 = "medium_buses_50"
    figure_name20 = "medium_buses_20"

    waiting_times_100 = filter_list(waiting_times_100)
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
    waiting_times_buses_20 = filter_list(waiting_times_buses_20)

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
    plt.errorbar(priority_factors, waiting_times_100, yerr=waiting_times_100_dev, color="limegreen", elinewidth=3,
                 alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_100, color="steelblue", marker='o', label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_100, yerr=waiting_times_cars_100_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_100, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_100, yerr=waiting_times_buses_100_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name100 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_70, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_70, yerr=waiting_times_70_dev, color="limegreen", elinewidth=3,
                 alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_70, color="steelblue", marker='o', label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_70, yerr=waiting_times_cars_70_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_70, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_70, yerr=waiting_times_buses_70_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name70 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_50, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_50, yerr=waiting_times_50_dev, color="limegreen", elinewidth=3,
                 alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_50, color="steelblue", marker='o', label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_50, yerr=waiting_times_cars_50_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_50, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_50, yerr=waiting_times_buses_50_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name50 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_20, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_20, yerr=waiting_times_20_dev, color="limegreen", elinewidth=3,
                 alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_20, color="steelblue", marker='o', label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_20, yerr=waiting_times_cars_20_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_20, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_20, yerr=waiting_times_buses_20_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name20 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_100_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_100_smooth, yerr=waiting_times_100_dev, color="limegreen",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_100_smooth, color="steelblue", marker='o',
             label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_100_smooth, yerr=waiting_times_cars_100_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_100_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_100_smooth, yerr=waiting_times_buses_100_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name100 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_70_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_70_smooth, yerr=waiting_times_70_dev, color="limegreen",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_70_smooth, color="steelblue", marker='o',
             label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_70_smooth, yerr=waiting_times_cars_70_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_70_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_70_smooth, yerr=waiting_times_buses_70_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name70 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_50_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_50_smooth, yerr=waiting_times_50_dev, color="limegreen",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_50_smooth, color="steelblue", marker='o',
             label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_50_smooth, yerr=waiting_times_cars_50_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_50_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_50_smooth, yerr=waiting_times_buses_50_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name50 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_20_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.errorbar(priority_factors, waiting_times_20_smooth, yerr=waiting_times_20_dev, color="limegreen",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_cars_20_smooth, color="steelblue", marker='o',
             label="Vehicles except buses")
    plt.errorbar(priority_factors, waiting_times_cars_20_smooth, yerr=waiting_times_cars_20_dev, color="steelblue",
                 elinewidth=3, alpha=0.4)
    plt.plot(priority_factors, waiting_times_buses_20_smooth, color="gold", marker='o', label="Buses")
    plt.errorbar(priority_factors, waiting_times_buses_20_smooth, yerr=waiting_times_buses_20_dev, color="gold",
                 elinewidth=3, alpha=0.4)
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/waiting_time/uniform/" + figure_name20 + "_smooth.png")
    plt.show()
