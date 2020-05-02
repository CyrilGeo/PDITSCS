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
    waiting_times_50 = []
    waiting_times_cars_50 = []
    waiting_times_buses_50 = []
    waiting_times_20 = []
    waiting_times_cars_20 = []
    waiting_times_buses_20 = []
    waiting_time_baseline = 0

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf1_buses_baseline/events.out.tfevents.1586625892.PC-CYRIL-LINUX.16867.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_time_baseline = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf1_buses_100/events.out.tfevents.1586626530.PC-CYRIL-LINUX.17259.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf2_buses_100/events.out.tfevents.1586627055.PC-CYRIL-LINUX.17520.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf3_buses_100/events.out.tfevents.1586945037.PC-CYRIL-LINUX.14327.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf4_buses_100/events.out.tfevents.1586945586.PC-CYRIL-LINUX.14862.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf5_buses_100/events.out.tfevents.1586808679.PC-CYRIL-LINUX.16131.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf6_buses_100/events.out.tfevents.1586945767.PC-CYRIL-LINUX.14934.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf7_buses_100/events.out.tfevents.1587117248.PC-CYRIL-LINUX.4337.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf8_buses_100/events.out.tfevents.1586946319.PC-CYRIL-LINUX.15152.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf9_buses_100/events.out.tfevents.1586946656.PC-CYRIL-LINUX.15261.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf10_buses_100/events.out.tfevents.1586629243.PC-CYRIL-LINUX.17975.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf11_buses_100/events.out.tfevents.1586947039.PC-CYRIL-LINUX.15413.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf12_buses_100/events.out.tfevents.1586947501.PC-CYRIL-LINUX.15601.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf13_buses_100/events.out.tfevents.1587117354.PC-CYRIL-LINUX.4380.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf14_buses_100/events.out.tfevents.1586948370.PC-CYRIL-LINUX.16005.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf15_buses_100/events.out.tfevents.1586949172.PC-CYRIL-LINUX.16309.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_100.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_100.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf1_buses_50/events.out.tfevents.1586626813.PC-CYRIL-LINUX.17326.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf2_buses_50/events.out.tfevents.1586627143.PC-CYRIL-LINUX.17553.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf3_buses_50/events.out.tfevents.1586945108.PC-CYRIL-LINUX.14365.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf4_buses_50/events.out.tfevents.1587117432.PC-CYRIL-LINUX.4430.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf5_buses_50/events.out.tfevents.1586629058.PC-CYRIL-LINUX.17899.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf6_buses_50/events.out.tfevents.1586945867.PC-CYRIL-LINUX.14962.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf7_buses_50/events.out.tfevents.1586946138.PC-CYRIL-LINUX.15075.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf8_buses_50/events.out.tfevents.1586946408.PC-CYRIL-LINUX.15181.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf9_buses_50/events.out.tfevents.1586946728.PC-CYRIL-LINUX.15290.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf10_buses_50/events.out.tfevents.1587117511.PC-CYRIL-LINUX.4501.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf11_buses_50/events.out.tfevents.1586947274.PC-CYRIL-LINUX.15526.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf12_buses_50/events.out.tfevents.1586947601.PC-CYRIL-LINUX.15637.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf13_buses_50/events.out.tfevents.1586948127.PC-CYRIL-LINUX.15826.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf14_buses_50/events.out.tfevents.1586948621.PC-CYRIL-LINUX.16050.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf15_buses_50/events.out.tfevents.1587117588.PC-CYRIL-LINUX.4535.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_50.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_50.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf1_buses_20/events.out.tfevents.1586626927.PC-CYRIL-LINUX.17409.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf2_buses_20/events.out.tfevents.1586627222.PC-CYRIL-LINUX.17582.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf3_buses_20/events.out.tfevents.1586945323.PC-CYRIL-LINUX.14642.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf4_buses_20/events.out.tfevents.1586945690.PC-CYRIL-LINUX.14901.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf5_buses_20/events.out.tfevents.1586629156.PC-CYRIL-LINUX.17934.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf6_buses_20/events.out.tfevents.1586945951.PC-CYRIL-LINUX.14997.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf7_buses_20/events.out.tfevents.1586946231.PC-CYRIL-LINUX.15115.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf8_buses_20/events.out.tfevents.1587117660.PC-CYRIL-LINUX.4573.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf9_buses_20/events.out.tfevents.1586946885.PC-CYRIL-LINUX.15363.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf10_buses_20/events.out.tfevents.1586808764.PC-CYRIL-LINUX.16191.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf11_buses_20/events.out.tfevents.1586947359.PC-CYRIL-LINUX.15557.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf12_buses_20/events.out.tfevents.1586947701.PC-CYRIL-LINUX.15700.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf13_buses_20/events.out.tfevents.1586948236.PC-CYRIL-LINUX.15946.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf14_buses_20/events.out.tfevents.1586949008.PC-CYRIL-LINUX.16123.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pf15_buses_20/events.out.tfevents.1587198282.PC-CYRIL-LINUX.12269.0"):
        for value in event.summary.value:
            if value.tag == "Average_waiting_time":
                waiting_times_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_cars":
                waiting_times_cars_20.append(value.simple_value)
            elif value.tag == "Average_waiting_time_buses":
                waiting_times_buses_20.append(value.simple_value)

    priority_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    figure_name100 = "low_buses_100"
    figure_name50 = "low_buses_50"
    figure_name20 = "low_buses_20"

    waiting_times_100 = filter_list(waiting_times_100)
    waiting_times_50 = filter_list(waiting_times_50)
    waiting_times_20 = filter_list(waiting_times_20)
    waiting_times_cars_100 = filter_list(waiting_times_cars_100)
    waiting_times_cars_50 = filter_list(waiting_times_cars_50)
    waiting_times_cars_20 = filter_list(waiting_times_cars_20)
    waiting_times_buses_100 = filter_list(waiting_times_buses_100)
    waiting_times_buses_50 = filter_list(waiting_times_buses_50)
    waiting_times_buses_20 = filter_list(waiting_times_buses_20)

    waiting_times_100_smooth = exponential_moving_average(waiting_times_100, 2)
    waiting_times_50_smooth = exponential_moving_average(waiting_times_50, 2)
    waiting_times_20_smooth = exponential_moving_average(waiting_times_20, 2)
    waiting_times_cars_100_smooth = exponential_moving_average(waiting_times_cars_100, 2)
    waiting_times_cars_50_smooth = exponential_moving_average(waiting_times_cars_50, 2)
    waiting_times_cars_20_smooth = exponential_moving_average(waiting_times_cars_20, 2)
    waiting_times_buses_100_smooth = exponential_moving_average(waiting_times_buses_100, 2)
    waiting_times_buses_50_smooth = exponential_moving_average(waiting_times_buses_50, 2)
    waiting_times_buses_20_smooth = exponential_moving_average(waiting_times_buses_20, 2)

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_100, color="limegreen", marker='o', label="Overall performance")
    plt.plot(priority_factors, waiting_times_cars_100, color="steelblue", marker='o', label="Vehicles except buses")
    plt.plot(priority_factors, waiting_times_buses_100, color="gold", marker='o', label="Buses")
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/uniform/" + figure_name100 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_50, color="limegreen", marker='o', label="Overall performance")
    plt.plot(priority_factors, waiting_times_cars_50, color="steelblue", marker='o', label="Cars")
    plt.plot(priority_factors, waiting_times_buses_50, color="gold", marker='o', label="Buses")
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/uniform/" + figure_name50 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_20, color="limegreen", marker='o', label="Overall performance")
    plt.plot(priority_factors, waiting_times_cars_20, color="steelblue", marker='o', label="Cars")
    plt.plot(priority_factors, waiting_times_buses_20, color="gold", marker='o', label="Buses")
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/uniform/" + figure_name20 + ".png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_100_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.plot(priority_factors, waiting_times_cars_100_smooth, color="steelblue", marker='o', label="Cars")
    plt.plot(priority_factors, waiting_times_buses_100_smooth, color="gold", marker='o', label="Buses")
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/uniform/" + figure_name100 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_50_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.plot(priority_factors, waiting_times_cars_50_smooth, color="steelblue", marker='o', label="Cars")
    plt.plot(priority_factors, waiting_times_buses_50_smooth, color="gold", marker='o', label="Buses")
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/uniform/" + figure_name50 + "_smooth.png")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(priority_factors, waiting_times_20_smooth, color="limegreen", marker='o', label="Overall performance")
    plt.plot(priority_factors, waiting_times_cars_20_smooth, color="steelblue", marker='o', label="Cars")
    plt.plot(priority_factors, waiting_times_buses_20_smooth, color="gold", marker='o', label="Buses")
    plt.axhline(y=waiting_time_baseline, color="r", label="Fixed time (10s)")
    plt.xlabel("Priority factor")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/uniform/" + figure_name20 + "_smooth.png")
    plt.show()
