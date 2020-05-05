import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    episodes = []
    rewards1 = []
    rewards2 = []
    rewards3 = []
    rewards4 = []
    rewards5 = []
    rewards6 = []
    rewards7 = []
    rewards8 = []
    rewards9 = []
    rewards10 = []
    rewards11 = []
    rewards12 = []
    rewards13 = []
    rewards14 = []
    rewards15 = []
    rewards16 = []
    rewards17 = []
    rewards18 = []
    rewards19 = []
    rewards20 = []
    rewards21 = []
    rewards22 = []
    rewards23 = []
    rewards24 = []
    rewards25 = []
    rewards26 = []
    rewards27 = []
    rewards28 = []
    waiting_times1 = []
    waiting_times2 = []
    waiting_times3 = []
    waiting_times4 = []
    waiting_times5 = []
    waiting_times6 = []
    waiting_times7 = []
    waiting_times8 = []
    waiting_times9 = []
    waiting_times10 = []
    waiting_times11 = []
    waiting_times12 = []
    waiting_times13 = []
    waiting_times14 = []
    waiting_times15 = []
    waiting_times16 = []
    waiting_times17 = []
    waiting_times18 = []
    waiting_times19 = []
    waiting_times20 = []
    waiting_times21 = []
    waiting_times22 = []
    waiting_times23 = []
    waiting_times24 = []
    waiting_times25 = []
    waiting_times26 = []
    waiting_times27 = []
    waiting_times28 = []
    rewards1_dev = []
    rewards2_dev = []
    rewards3_dev = []
    rewards4_dev = []
    rewards5_dev = []
    rewards6_dev = []
    rewards7_dev = []
    rewards8_dev = []
    rewards9_dev = []
    rewards10_dev = []
    rewards11_dev = []
    rewards12_dev = []
    rewards13_dev = []
    rewards14_dev = []
    rewards15_dev = []
    rewards16_dev = []
    rewards17_dev = []
    rewards18_dev = []
    rewards19_dev = []
    rewards20_dev = []
    rewards21_dev = []
    rewards22_dev = []
    rewards23_dev = []
    rewards24_dev = []
    rewards25_dev = []
    rewards26_dev = []
    rewards27_dev = []
    rewards28_dev = []
    waiting_times1_dev = []
    waiting_times2_dev = []
    waiting_times3_dev = []
    waiting_times4_dev = []
    waiting_times5_dev = []
    waiting_times6_dev = []
    waiting_times7_dev = []
    waiting_times8_dev = []
    waiting_times9_dev = []
    waiting_times10_dev = []
    waiting_times11_dev = []
    waiting_times12_dev = []
    waiting_times13_dev = []
    waiting_times14_dev = []
    waiting_times15_dev = []
    waiting_times16_dev = []
    waiting_times17_dev = []
    waiting_times18_dev = []
    waiting_times19_dev = []
    waiting_times20_dev = []
    waiting_times21_dev = []
    waiting_times22_dev = []
    waiting_times23_dev = []
    waiting_times24_dev = []
    waiting_times25_dev = []
    waiting_times26_dev = []
    waiting_times27_dev = []
    waiting_times28_dev = []
    baseline_r1 = 0
    baseline_w1 = 0
    baseline_r_dev1 = 0
    baseline_w_dev1 = 0
    baseline_r2 = 0
    baseline_w2 = 0
    baseline_r_dev2 = 0
    baseline_w_dev2 = 0
    baseline_r3 = 0
    baseline_w3 = 0
    baseline_r_dev3 = 0
    baseline_w_dev3 = 0
    baseline_r4 = 0
    baseline_w4 = 0
    baseline_r_dev4 = 0
    baseline_w_dev4 = 0
    baseline_r5 = 0
    baseline_w5 = 0
    baseline_r_dev5 = 0
    baseline_w_dev5 = 0
    baseline_r6 = 0
    baseline_w6 = 0
    baseline_r_dev6 = 0
    baseline_w_dev6 = 0
    baseline_r7 = 0
    baseline_w7 = 0
    baseline_r_dev7 = 0
    baseline_w_dev7 = 0
    baseline_adapted_r1 = 0
    baseline_adapted_w1 = 0
    baseline_adapted_r_dev1 = 0
    baseline_adapted_w_dev1 = 0
    baseline_adapted_r2 = 0
    baseline_adapted_w2 = 0
    baseline_adapted_r_dev2 = 0
    baseline_adapted_w_dev2 = 0
    baseline_adapted_r3 = 0
    baseline_adapted_w3 = 0
    baseline_adapted_r_dev3 = 0
    baseline_adapted_w_dev3 = 0

    figure_location = "LuST/"
    figure_name = "training"

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over300_baseline/events.out.tfevents.1587729335.PC-CYRIL-LINUX.11597.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r1 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w1 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev1 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev1 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_low/events.out.tfevents.1587749218.alan-compute-05.4856.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards1.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times1.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards1_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times1_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_low/events.out.tfevents.1587751142.alan-compute-06.24637.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards2.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times2.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards2_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times2_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_low/events.out.tfevents.1587751666.alan-compute-01.2195.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards3.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times3.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards3_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times3_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_low/events.out.tfevents.1587751672.alan-compute-01.2239.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards4.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times4.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards4_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times4_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over60_baseline/events.out.tfevents.1587729370.PC-CYRIL-LINUX.11624.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r2 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w2 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev2 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev2 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_medium/events.out.tfevents.1587741096.alan-compute-09.48426.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards5.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times5.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards5_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times5_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_medium/events.out.tfevents.1587744205.alan-compute-05.4406.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards6.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times6.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards6_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times6_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_medium/events.out.tfevents.1587744724.alan-compute-01.21540.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards7.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times7.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards7_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times7_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_medium/events.out.tfevents.1587744724.alan-compute-01.21539.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards8.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times8.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards8_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times8_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over30_baseline/events.out.tfevents.1587729414.PC-CYRIL-LINUX.11651.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r3 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w3 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev3 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev3 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_high/events.out.tfevents.1587751455.alan-compute-06.25560.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards9.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times9.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards9_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times9_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_high/events.out.tfevents.1587751455.alan-compute-06.25560.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards10.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times10.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards10_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times10_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_high/events.out.tfevents.1587751984.alan-compute-08.110660.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards11.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times11.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards11_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times11_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_high/events.out.tfevents.1587752346.alan-compute-08.116992.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards12.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times12.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards12_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times12_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over60_ver1over300_baseline/events.out.tfevents.1587733610.PC-CYRIL-LINUX.12865.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r4 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w4 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev4 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev4 = value.simple_value

    # Adapted baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over60_ver1over300_adapted/events.out.tfevents.1587734319.PC-CYRIL-LINUX.13126.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_adapted_r1 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_adapted_w1 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_adapted_r_dev1 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_adapted_w_dev1 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_60_300_100/events.out.tfevents.1587805482.alan-compute-01.4722.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards13.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times13.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards13_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times13_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_60_300_70/events.out.tfevents.1587805385.alan-compute-02.12474.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards14.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times14.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards14_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times14_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_60_300_50/events.out.tfevents.1587805511.alan-compute-02.13166.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards15.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times15.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards15_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times15_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_60_300_20/events.out.tfevents.1587805581.alan-compute-02.13549.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards16.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times16.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards16_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times16_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over30_ver1over300_baseline/events.out.tfevents.1587733562.PC-CYRIL-LINUX.12837.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r5 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w5 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev5 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev5 = value.simple_value

    # Adapted baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over30_ver1over300_adapted/events.out.tfevents.1587734900.PC-CYRIL-LINUX.13460.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_adapted_r2 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_adapted_w2 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_adapted_r_dev2 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_adapted_w_dev2 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_300_100/events.out.tfevents.1587805598.alan-compute-02.13823.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards17.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times17.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards17_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times17_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_300_70/events.out.tfevents.1587805696.alan-compute-03.8817.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards18.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times18.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards18_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times18_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_300_50/events.out.tfevents.1587805783.alan-compute-03.10186.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards19.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times19.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards19_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times19_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_300_20/events.out.tfevents.1587805835.alan-compute-04.22155.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards20.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times20.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards20_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times20_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over30_ver1over60_baseline/events.out.tfevents.1587733504.PC-CYRIL-LINUX.12795.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r6 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w6 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev6 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev6 = value.simple_value

    # Adapted baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/hor1over30_ver1over60_adapted/events.out.tfevents.1587735610.PC-CYRIL-LINUX.13824.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_adapted_r3 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_adapted_w3 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_adapted_r_dev3 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_adapted_w_dev3 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_60_100/events.out.tfevents.1587806613.alan-compute-05.9374.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards21.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times21.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards21_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times21_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_60_70/events.out.tfevents.1587806661.alan-compute-05.9408.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards22.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times22.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards22_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times22_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_60_50/events.out.tfevents.1587806732.alan-compute-06.168231.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards23.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times23.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards23_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times23_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_hor_30_60_20/events.out.tfevents.1587806772.alan-compute-06.168371.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards24.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times24.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards24_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times24_dev.append(value.simple_value)

    # Baseline
    for event in tf.compat.v1.train.summary_iterator(
            "runs/uniform_1over15_baseline/events.out.tfevents.1587729529.PC-CYRIL-LINUX.11671.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                baseline_r7 = value.simple_value
            elif value.tag == "Average_waiting_time":
                baseline_w7 = value.simple_value
            elif value.tag == "Reward_standard_deviation":
                baseline_r_dev7 = value.simple_value
            elif value.tag == "Waiting_time_standard_deviation":
                baseline_w_dev7 = value.simple_value

    # 100% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_100_veryhigh/events.out.tfevents.1587752514.alan-compute-08.119841.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards25.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times25.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards25_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times25_dev.append(value.simple_value)

    # 70% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_70_veryhigh/events.out.tfevents.1587754019.alan-compute-09.91306.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards26.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times26.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards26_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times26_dev.append(value.simple_value)

    # 50% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_50_veryhigh/events.out.tfevents.1587753838.alan-compute-05.5294.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards27.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times27.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards27_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times27_dev.append(value.simple_value)

    # 20% detection rate
    for event in tf.compat.v1.train.summary_iterator(
            "runs/model_20_veryhigh/events.out.tfevents.1587755872.alan-compute-06.36963.0"):
        for value in event.summary.value:
            if value.tag == "Average_reward":
                rewards28.append(value.simple_value)
            elif value.tag == "Average_waiting_time":
                waiting_times28.append(value.simple_value)
            elif value.tag == "Reward_standard_deviation":
                rewards28_dev.append(value.simple_value)
            elif value.tag == "Waiting_time_standard_deviation":
                waiting_times28_dev.append(value.simple_value)

    bar_width = 0.15
    bars1 = [baseline_r1, baseline_r2, baseline_r3, baseline_r7, baseline_adapted_r1, baseline_adapted_r2,
             baseline_adapted_r3]
    bars2 = [rewards4[-1], rewards8[-1], rewards12[-1], rewards28[-1], rewards16[-1], rewards20[-1], rewards24[-1]]
    bars3 = [rewards3[-1], rewards7[-1], rewards11[-1], rewards27[-1], rewards15[-1], rewards19[-1], rewards23[-1]]
    bars4 = [rewards2[-1], rewards6[-1], rewards10[-1], rewards26[-1], rewards14[-1], rewards18[-1], rewards22[-1]]
    bars5 = [rewards1[-1], rewards5[-1], rewards9[-1], rewards25[-1], rewards13[-1], rewards17[-1], rewards21[-1]]
    err1 = [baseline_r_dev1, baseline_r_dev2, baseline_r_dev3, baseline_r_dev7, baseline_adapted_r_dev1,
            baseline_adapted_r_dev2, baseline_adapted_r_dev3]
    err2 = [rewards4_dev[-1], rewards8_dev[-1], rewards12_dev[-1], rewards28_dev[-1], rewards16_dev[-1],
            rewards20_dev[-1], rewards24_dev[-1]]
    err3 = [rewards3_dev[-1], rewards7_dev[-1], rewards11_dev[-1], rewards27_dev[-1], rewards15_dev[-1],
            rewards19_dev[-1], rewards23_dev[-1]]
    err4 = [rewards2_dev[-1], rewards6_dev[-1], rewards10_dev[-1], rewards26_dev[-1], rewards14_dev[-1],
            rewards18_dev[-1], rewards22_dev[-1]]
    err5 = [rewards1_dev[-1], rewards5_dev[-1], rewards9_dev[-1], rewards25_dev[-1], rewards13_dev[-1],
            rewards17_dev[-1], rewards21_dev[-1]]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    plt.figure(figsize=[10, 4.8])
    plt.grid(axis="y")
    plt.bar(r1, bars1, color="r", width=bar_width, edgecolor="white", label="Fixed time", yerr=err1,
            capsize=3)
    plt.bar(r2, bars2, color="gold", width=bar_width, edgecolor="white", label="20% detection rate", yerr=err2,
            capsize=3)
    plt.bar(r3, bars3, color="darkorange", width=bar_width, edgecolor="white", label="50% detection rate", yerr=err3,
            capsize=3)
    plt.bar(r4, bars4, color="steelblue", width=bar_width, edgecolor="white", label="70% detection rate", yerr=err4,
            capsize=3)
    plt.bar(r5, bars5, color="limegreen", width=bar_width, edgecolor="white", label="100% detection rate", yerr=err5,
            capsize=3)
    plt.xticks([r + 2 * bar_width for r in range(len(bars1))],
               ["Low", "Medium", "High", "Very high", "Medium - Low", "High - Low", "High - Medium"])
    plt.ylabel("Average reward")
    plt.xlabel("Traffic flow")
    plt.legend()
    plt.savefig("figures/reward/flow_perf1.png")
    plt.show()

    bar_width = 0.15
    bars1 = [baseline_w1, baseline_w2, baseline_w3, baseline_w7, baseline_adapted_w1, baseline_adapted_w2,
             baseline_adapted_w3]
    bars2 = [waiting_times4[-1], waiting_times8[-1], waiting_times12[-1], waiting_times28[-1], waiting_times16[-1],
             waiting_times20[-1], waiting_times24[-1]]
    bars3 = [waiting_times3[-1], waiting_times7[-1], waiting_times11[-1], waiting_times27[-1], waiting_times15[-1],
             waiting_times19[-1], waiting_times23[-1]]
    bars4 = [waiting_times2[-1], waiting_times6[-1], waiting_times10[-1], waiting_times26[-1], waiting_times14[-1],
             waiting_times18[-1], waiting_times22[-1]]
    bars5 = [waiting_times1[-1], waiting_times5[-1], waiting_times9[-1], waiting_times25[-1], waiting_times13[-1],
             waiting_times17[-1], waiting_times21[-1]]
    err1 = [baseline_w_dev1, baseline_w_dev2, baseline_w_dev3, baseline_w_dev7, baseline_adapted_w_dev1,
            baseline_adapted_w_dev2, baseline_adapted_w_dev3]
    err2 = [waiting_times4_dev[-1], waiting_times8_dev[-1], waiting_times12_dev[-1], waiting_times28_dev[-1],
            waiting_times16_dev[-1], waiting_times20_dev[-1], waiting_times24_dev[-1]]
    err3 = [waiting_times3_dev[-1], waiting_times7_dev[-1], waiting_times11_dev[-1], waiting_times27_dev[-1],
            waiting_times15_dev[-1], waiting_times19_dev[-1], waiting_times23_dev[-1]]
    err4 = [waiting_times2_dev[-1], waiting_times6_dev[-1], waiting_times10_dev[-1], waiting_times26_dev[-1],
            waiting_times14_dev[-1], waiting_times18_dev[-1], waiting_times22_dev[-1]]
    err5 = [waiting_times1_dev[-1], waiting_times5_dev[-1], waiting_times9_dev[-1], waiting_times25_dev[-1],
            waiting_times13_dev[-1], waiting_times17_dev[-1], waiting_times21_dev[-1]]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    plt.figure(figsize=[10, 4.8])
    plt.grid(axis="y")
    plt.bar(r1, bars1, color="r", width=bar_width, edgecolor="white", label="Fixed time", yerr=err1,
            capsize=3)
    plt.bar(r2, bars2, color="gold", width=bar_width, edgecolor="white", label="20% detection rate", yerr=err2,
            capsize=3)
    plt.bar(r3, bars3, color="darkorange", width=bar_width, edgecolor="white", label="50% detection rate", yerr=err3,
            capsize=3)
    plt.bar(r4, bars4, color="steelblue", width=bar_width, edgecolor="white", label="70% detection rate", yerr=err4,
            capsize=3)
    plt.bar(r5, bars5, color="limegreen", width=bar_width, edgecolor="white", label="100% detection rate", yerr=err5,
            capsize=3)
    plt.xticks([r + 2 * bar_width for r in range(len(bars1))],
               ["Low", "Medium", "High", "Very high", "Medium - Low", "High - Low", "High - Medium"])
    plt.ylabel("Average waiting time (s)")
    plt.xlabel("Traffic flow")
    plt.legend()
    plt.savefig("figures/waiting_time/flow_perf1.png")
    plt.show()

    '''bar_width = 0.15
    bars1 = [baseline_r1, baseline_r2, baseline_r3, baseline_r4, baseline_r5, baseline_r6]
    bars2 = [rewards4[-1], rewards8[-1], rewards12[-1], baseline_adapted_r1, baseline_adapted_r2, baseline_adapted_r3]
    bars3 = [rewards3[-1], rewards7[-1], rewards11[-1], rewards16[-1], rewards20[-1], rewards24[-1]]
    bars4 = [rewards2[-1], rewards6[-1], rewards10[-1], rewards15[-1], rewards19[-1], rewards23[-1]]
    bars5 = [rewards1[-1], rewards5[-1], rewards9[-1], rewards14[-1], rewards18[-1], rewards22[-1]]
    bars6 = [rewards13[-1], rewards17[-1], rewards21[-1]]
    err1 = [baseline_r_dev1, baseline_r_dev2, baseline_r_dev3, baseline_r_dev4, baseline_r_dev5, baseline_r_dev6]
    err2 = [rewards4_dev[-1], rewards8_dev[-1], rewards12_dev[-1], baseline_adapted_r_dev1, baseline_adapted_r_dev2, baseline_adapted_r_dev3]
    err3 = [rewards3_dev[-1], rewards7_dev[-1], rewards11_dev[-1], rewards16_dev[-1], rewards20_dev[-1], rewards24_dev[-1]]
    err4 = [rewards2_dev[-1], rewards6_dev[-1], rewards10_dev[-1], rewards15_dev[-1], rewards19_dev[-1], rewards23_dev[-1]]
    err5 = [rewards1_dev[-1], rewards5_dev[-1], rewards9_dev[-1], rewards14_dev[-1], rewards18_dev[-1], rewards22_dev[-1]]
    err6 = [rewards13_dev[-1], rewards17_dev[-1], rewards21_dev[-1]]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5[3:]]
    plt.figure()
    plt.grid(axis="y")
    plt.bar(r1, bars1, color="r", width=bar_width, edgecolor="white", label="Fixed time (10s)", yerr=err1, capsize=3)
    plt.bar(r3, bars3, color=["gold", "gold", "gold", "darkviolet", "darkviolet", "darkviolet"], width=bar_width, edgecolor="white", label=["20% detection rate", "20% detection rate", "20% detection rate", "Adapted fixed time", "Adapted fixed time", "Adapted fixed time"], yerr=err3, capsize=3)
    plt.bar(r3, bars3, color=["darkorange", "darkorange", "darkorange", "gold", "gold", "gold"], width=bar_width, edgecolor="white", label=["50% detection rate", "50% detection rate", "50% detection rate", "20% detection rate", "20% detection rate", "20% detection rate"], yerr=err3, capsize=3)
    plt.bar(r4, bars4, color=["steelblue", "steelblue", "steelblue", "darkorange", "darkorange", "darkorange"], width=bar_width, edgecolor="white", label=["70% detection rate", "70% detection rate", "70% detection rate", "50% detection rate", "50% detection rate", "50% detection rate"], yerr=err4, capsize=3)
    plt.bar(r5, bars5, color=["limegreen", "limegreen", "limegreen", "steelblue", "steelblue", "steelblue"], width=bar_width, edgecolor="white", label=["100% detection rate", "100% detection rate", "100% detection rate", "70% detection rate", "70% detection rate", "70% detection rate"], yerr=err5, capsize=3)
    plt.bar(r6, bars6, color="limegreen", width=bar_width, edgecolor="white", label="100% detection rate", yerr=err6, capsize=3)
    plt.xticks([r + 2 * bar_width for r in range(len(bars1))],
               ["Low", "Medium", "High", "Medium - Low", "High - Low", "High - Medium"])
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/reward/flow_perf.png")
    plt.show()'''
