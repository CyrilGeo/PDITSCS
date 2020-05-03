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
    reward11 = 0
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
    waiting_time11 = 0
    waiting_time1_veh = 0
    waiting_time2_veh = 0
    waiting_time3_veh = 0
    waiting_time4_veh = 0
    waiting_time5_veh = 0
    waiting_time6_veh = 0
    waiting_time7_veh = 0
    waiting_time8_veh = 0
    waiting_time9_veh = 0
    waiting_time10_veh = 0
    waiting_time11_veh = 0
    waiting_time1_ped = 0
    waiting_time2_ped = 0
    waiting_time3_ped = 0
    waiting_time4_ped = 0
    waiting_time5_ped = 0
    waiting_time6_ped = 0
    waiting_time7_ped = 0
    waiting_time8_ped = 0
    waiting_time9_ped = 0
    waiting_time10_ped = 0
    waiting_time11_ped = 0
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
    reward11_dev = 0
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
    waiting_time11_dev = 0
    waiting_time1_veh_dev = 0
    waiting_time2_veh_dev = 0
    waiting_time3_veh_dev = 0
    waiting_time4_veh_dev = 0
    waiting_time5_veh_dev = 0
    waiting_time6_veh_dev = 0
    waiting_time7_veh_dev = 0
    waiting_time8_veh_dev = 0
    waiting_time9_veh_dev = 0
    waiting_time10_veh_dev = 0
    waiting_time11_veh_dev = 0
    waiting_time1_ped_dev = 0
    waiting_time2_ped_dev = 0
    waiting_time3_ped_dev = 0
    waiting_time4_ped_dev = 0
    waiting_time5_ped_dev = 0
    waiting_time6_ped_dev = 0
    waiting_time7_ped_dev = 0
    waiting_time8_ped_dev = 0
    waiting_time9_ped_dev = 0
    waiting_time10_ped_dev = 0
    waiting_time11_ped_dev = 0

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_pedestrian_baseline/events.out.tfevents.1588416498.PC-CYRIL-LINUX.32211.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward1 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time1 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward1_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time1_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time1_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time1_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time1_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time1_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_ped_neglected/events.out.tfevents.1588414921.PC-CYRIL-LINUX.31245.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward2 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time2 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward2_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time2_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time2_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time2_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time2_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time2_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian/events.out.tfevents.1588415266.PC-CYRIL-LINUX.31656.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward3 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time3 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward3_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time3_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time3_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time3_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time3_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time3_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian2/events.out.tfevents.1588415407.PC-CYRIL-LINUX.31711.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward4 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time4 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward4_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time4_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time4_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time4_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time4_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time4_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian3/events.out.tfevents.1588415524.PC-CYRIL-LINUX.31774.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward5 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time5 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward5_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time5_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time5_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time5_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time5_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time5_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian4/events.out.tfevents.1588415659.PC-CYRIL-LINUX.31807.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward6 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time6 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward6_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time6_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time6_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time6_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time6_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time6_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian5/events.out.tfevents.1588415771.PC-CYRIL-LINUX.31866.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward7 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time7 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward7_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time7_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time7_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time7_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time7_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time7_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian6/events.out.tfevents.1588415904.PC-CYRIL-LINUX.31902.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward8 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time8 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward8_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time8_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time8_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time8_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time8_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time8_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian7/events.out.tfevents.1588416043.PC-CYRIL-LINUX.31994.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward9 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time9 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward9_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time9_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time9_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time9_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time9_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time9_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_pedestrian8/events.out.tfevents.1588416205.PC-CYRIL-LINUX.32092.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward10 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time10 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward10_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time10_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time10_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time10_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time10_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time10_ped_dev = value.simple_value

    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_100_ped_perfect/events.out.tfevents.1588416364.PC-CYRIL-LINUX.32165.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                reward11 = value.simple_value
            elif value.tag == "Average_waiting_time":
                waiting_time11 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                reward11_dev = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_time11_dev = value.simple_value
            elif value.tag == "Average_waiting_time_vehicles":
                waiting_time11_veh = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_vehicles":
                waiting_time11_veh_dev = value.simple_value
            elif value.tag == "Average_waiting_time_pedestrians":
                waiting_time11_ped = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation_pedestrians":
                waiting_time11_ped_dev = value.simple_value

    figure_name = "figures/waiting_time/uniform/comp_pedestrian.png"

    bar_width = 0.25
    bars1 = [waiting_time1, waiting_time2, waiting_time3, waiting_time7, waiting_time8, waiting_time4, waiting_time5,
             waiting_time6, waiting_time9, waiting_time10, waiting_time11]
    bars2 = [waiting_time1_veh, waiting_time2_veh, waiting_time3_veh, waiting_time7_veh, waiting_time8_veh,
             waiting_time4_veh, waiting_time5_veh, waiting_time6_veh, waiting_time9_veh, waiting_time10_veh,
             waiting_time11_veh]
    bars3 = [waiting_time1_ped, waiting_time2_ped, waiting_time3_ped, waiting_time7_ped, waiting_time8_ped,
             waiting_time4_ped, waiting_time5_ped, waiting_time6_ped, waiting_time9_ped, waiting_time10_ped,
             waiting_time11_ped]
    err1 = [waiting_time1_dev, waiting_time2_dev, waiting_time3_dev, waiting_time7_dev, waiting_time8_dev,
            waiting_time4_dev, waiting_time5_dev, waiting_time6_dev, waiting_time9_dev, waiting_time10_dev,
            waiting_time11_dev]
    err2 = [waiting_time1_veh_dev, waiting_time2_veh_dev, waiting_time3_veh_dev, waiting_time7_veh_dev,
            waiting_time8_veh_dev, waiting_time4_veh_dev, waiting_time5_veh_dev, waiting_time6_veh_dev,
            waiting_time9_veh_dev, waiting_time10_veh_dev, waiting_time11_veh_dev]
    err3 = [waiting_time1_ped_dev, waiting_time2_ped_dev, waiting_time3_ped_dev, waiting_time7_ped_dev,
            waiting_time8_ped_dev, waiting_time4_ped_dev, waiting_time5_ped_dev, waiting_time6_ped_dev,
            waiting_time9_ped_dev, waiting_time10_ped_dev, waiting_time11_ped_dev]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    plt.figure()
    plt.grid(axis="y")
    plt.bar(r1, bars1, color="limegreen", width=bar_width, edgecolor="white", label="Overall performance", yerr=err1,
            capsize=2)
    plt.bar(r2, bars2, color="steelblue", width=bar_width, edgecolor="white", label="Vehicles", yerr=err2, capsize=2)
    plt.bar(r3, bars3, color="gold", width=bar_width, edgecolor="white", label="Pedestrians", yerr=err3,
            capsize=2)
    plt.xticks([r + bar_width for r in range(len(bars1))],
               ["Fixed", "Neg.", "Zo1", "Zo2", "Zo3", "Cr1", "Cr2", "Cr3", "Cr4", "Cr5", "Perfect"])
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig(figure_name)
    plt.show()
