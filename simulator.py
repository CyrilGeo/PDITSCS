"""
Simulator of the simple intersection.
"""

import os
import sys
import optparse
import random
import statistics

# Importation of python modules from the SUMO_HOME/tools library (importations of sumolib and traci must be placed after
# this)
# Set the environment variable using: export SUMO_HOME='/usr/share/sumo'
# To know where sumo directory is: whereis sumo
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Environment variable 'SUMO_HOME' has to be declared.")

from sumolib import checkBinary  # Checks for the binary in environ vars

# For parallel use uncomment first line, for GUI use uncomment second line
import libsumo as traci


# import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="Run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


class Simulator:
    """
    The simulator.
    """

    def __init__(self, nb_episodes, nb_episode_steps, detection_rate, min_phase_duration, route_probs, hour_of_the_day,
                 gui=False, hourly_probs=None):
        self.N = nb_episodes
        self.n = nb_episode_steps
        self.episodeCnt = 1
        self.detectionRate = detection_rate
        self.minPhaseDuration = min_phase_duration
        self.hourOfTheDay = hour_of_the_day
        self.currNbIterations = 0
        self.currPhaseTime = None
        self.routeProbs = route_probs
        self.detectedColor = "0, 255, 0"
        self.undetectedColor = "255, 0, 0"
        self.laneIDs = ["in_west_0", "in_north_0", "in_east_0", "in_south_0"]
        self.sumoBinary = None
        self.episodeEnd = 1  # 1 if last step of an episode, 0 otherwise
        self.hourlyProbs = hourly_probs
        self.job_id = "0"
        if "SLURM_JOB_ID" in os.environ:
            self.job_id = os.environ["SLURM_JOB_ID"]

        # State variables
        self.detectedCarCnt = None
        self.distanceNearestDetectedVeh = None
        self.normCurrPhaseTime = None
        self.amberPhase = None
        self.currDayTime = None

        # Stats
        self.episodes = []
        self.reward = None
        self.rewards = []
        self.averageRewards = []
        self.cumWaitingTime = 0
        self.cumWaitingTimeDetected = 0
        self.cumWaitingTimeUndetected = 0
        self.nbGeneratedVeh = 0
        self.nbGeneratedVehDetected = 0
        self.nbGeneratedVehUndetected = 0
        self.averageWaitingTimes = []
        self.averageWaitingTimesDetected = []
        self.averageWaitingTimesUndetected = []

        # Determines whether to use the simulator's GUI or not
        options = get_options()
        if gui:
            if options.nogui:
                self.sumoBinary = checkBinary("sumo")
            else:
                self.sumoBinary = checkBinary("sumo-gui")
        else:
            self.sumoBinary = checkBinary("sumo")

        with open("sumo_sim/simple_intersection_" + self.job_id + ".sumocfg", "w") as config:
            print("""<configuration>
    <input>
        <net-file value="simple_intersection.net.xml"/>
        <route-files value="simple_intersection_""" + self.job_id + """.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value=""" + '"' + str(self.n) + '"' + """/>
    </time>
</configuration>""", file=config)

        self.init_new_episode()

        self.defaultDistances = [-traci.lane.getLength(x) for x in self.laneIDs]
        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.update_state(veh_ids)

    def init_new_episode(self):
        """
        Initializes a new episode.
        :return: None
        """
        print("LOADING NEW EPISODE")
        self.generate_traffic(self.hourlyProbs)

        # Starting sumo as a subprocess
        traci.start([self.sumoBinary, "-c", "sumo_sim/simple_intersection_" + self.job_id + ".sumocfg"])
        '''traci.start(
            [self.sumoBinary, "-c", "sumo_sim/simple_intersection_" + self.job_id + ".sumocfg", "--tripinfo-output",
             "tripinfo_" + self.job_id + ".xml"])'''

    def generate_traffic(self, hourly_probs=None):
        """
        Randomly generates the route file that determines the traffic in the simulation.
        :param hourly_probs: possible predetermined hourly car generation probabilities for each incoming road
        :return: None
        """
        random.seed()

        with open("sumo_sim/simple_intersection_" + self.job_id + ".rou.xml", "w") as routes:
            print("""<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" maxSpeed="55.55" length="4.5"/>

    <route id="route0" edges="in_west out_north"/>
    <route id="route1" edges="in_west out_east"/>
    <route id="route2" edges="in_west out_south"/>
    <route id="route3" edges="in_north out_east"/>
    <route id="route4" edges="in_north out_south"/>
    <route id="route5" edges="in_north out_west"/>
    <route id="route6" edges="in_east out_south"/>
    <route id="route7" edges="in_east out_west"/>
    <route id="route8" edges="in_east out_north"/>
    <route id="route9" edges="in_south out_west"/>
    <route id="route10" edges="in_south out_north"/>
    <route id="route11" edges="in_south out_east"/>""", file=routes)

            # Randomly choosing if a vehicle is generated for each step and each route
            if hourly_probs:
                nb_configurations = len(hourly_probs)
                k = -1
                for i in range(self.n):
                    if i % 3600 == 0:
                        k = (k + 1) % nb_configurations
                    for j in range(len(hourly_probs[k])):
                        if random.uniform(0, 1) < hourly_probs[k][j]:
                            veh_color = self.select_color()
                            print('    <vehicle id="' + str(self.nbGeneratedVeh) + '" type="car" route="route' + str(
                                j) + '" depart="' + str(
                                i) + '" color="' + veh_color + '" departSpeed="max"/>', file=routes)
                            self.nbGeneratedVeh += 1
                            if veh_color == self.detectedColor:
                                self.nbGeneratedVehDetected += 1
                            else:
                                self.nbGeneratedVehUndetected += 1
            else:
                for i in range(self.n):
                    for j in range(len(self.routeProbs)):
                        if random.uniform(0, 1) < self.routeProbs[j]:
                            veh_color = self.select_color()
                            print('    <vehicle id="' + str(self.nbGeneratedVeh) + '" type="car" route="route' + str(
                                j) + '" depart="' + str(
                                i) + '" color="' + veh_color + '" departSpeed="max"/>', file=routes)
                            self.nbGeneratedVeh += 1
                            if veh_color == self.detectedColor:
                                self.nbGeneratedVehDetected += 1
                            else:
                                self.nbGeneratedVehUndetected += 1

            print("</routes>", file=routes)

    def select_color(self):
        """
        Randomly chooses if a generated vehicle is detected or not.
        :return: None
        """
        if random.uniform(0, 1) < self.detectionRate:
            return self.detectedColor
        return self.undetectedColor

    def step(self, action=None):
        """
        Performs one iteration/step in the simulator (environment).
        :param action: the action to perform
        :return: True if simulation is to be continued, False otherwise
        """
        self.episodeEnd = 0

        # Episode stops when all raw files have been exhausted (no vehicles left in the simulation)
        if traci.simulation.getMinExpectedNumber() <= 0:
            traci.close()
            print("EPISODE", self.episodeCnt, "DONE")
            average_reward = statistics.mean(self.rewards)
            print("Average reward:", average_reward)
            self.episodes.append(self.episodeCnt)
            self.averageRewards.append(average_reward)
            average_waiting_time = self.cumWaitingTime / self.nbGeneratedVeh if self.nbGeneratedVeh != 0 else 0
            average_waiting_time_detected = self.cumWaitingTimeDetected / self.nbGeneratedVehDetected if self.nbGeneratedVehDetected != 0 else 0
            average_waiting_time_undetected = self.cumWaitingTimeUndetected / self.nbGeneratedVehUndetected if self.nbGeneratedVehUndetected != 0 else 0
            print("Average waiting time:", average_waiting_time)
            print("Average waiting time for detected vehicles:", average_waiting_time_detected)
            print("Average waiting time for undetected vehicles:", average_waiting_time_undetected)
            self.averageWaitingTimes.append(average_waiting_time)
            self.averageWaitingTimesDetected.append(average_waiting_time_detected)
            self.averageWaitingTimesUndetected.append(average_waiting_time_undetected)
            self.rewards.clear()
            self.cumWaitingTime = 0
            self.cumWaitingTimeDetected = 0
            self.cumWaitingTimeUndetected = 0
            self.nbGeneratedVeh = 0
            self.nbGeneratedVehDetected = 0
            self.nbGeneratedVehUndetected = 0

            if self.N:
                if self.episodeCnt < self.N:
                    self.episodeCnt += 1
                    self.init_new_episode()
                    self.episodeEnd = 1
                else:
                    print("END OF SIMULATION")
                    return False
            else:
                self.episodeCnt += 1
                self.init_new_episode()
                self.episodeEnd = 1

        # Action decided by the value given in argument
        if action is not None:
            if action == 1 and self.currPhaseTime >= self.minPhaseDuration:
                self.next_phase()

        # Fixed phase duration
        elif self.currPhaseTime >= 10:
            self.next_phase()

        # Adapted fixed phase duration
        '''elif (self.currPhaseTime >= 10 and traci.trafficlight.getPhase("center") == 0) or (
                self.currPhaseTime >= 20 and traci.trafficlight.getPhase("center") == 2):
            self.next_phase()'''

        traci.simulationStep()
        self.currNbIterations += 1
        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.update_state(veh_ids)
        self.update_reward(veh_ids)
        self.increment_waiting_time(veh_ids)
        self.rewards.append(self.reward)
        return True

    @staticmethod
    def next_phase():
        """
        Switches traffic lights to the next phase.
        :return: None
        """
        traci.trafficlight.setPhase("center", (traci.trafficlight.getPhase("center") + 1) % 4)

    def update_state(self, veh_ids):
        """
        Updates the state representation.
        :param veh_ids: the IDs of the vehicles currently simulated
        :return: None
        """
        self.detectedCarCnt = self.count_detected_veh(veh_ids)
        self.distanceNearestDetectedVeh = [-x / y for x, y in zip(self.get_distances(veh_ids), self.defaultDistances)]
        current_phase = traci.trafficlight.getPhase("center")
        if current_phase == 0:
            self.detectedCarCnt[1] = -self.detectedCarCnt[1]
            self.detectedCarCnt[3] = -self.detectedCarCnt[3]
            self.distanceNearestDetectedVeh[1] = -self.distanceNearestDetectedVeh[1]
            self.distanceNearestDetectedVeh[3] = -self.distanceNearestDetectedVeh[3]
        elif current_phase == 2:
            self.detectedCarCnt[0] = -self.detectedCarCnt[0]
            self.detectedCarCnt[2] = -self.detectedCarCnt[2]
            self.distanceNearestDetectedVeh[0] = -self.distanceNearestDetectedVeh[0]
            self.distanceNearestDetectedVeh[2] = -self.distanceNearestDetectedVeh[2]

        self.currPhaseTime = (traci.simulation.getTime() + traci.trafficlight.getPhaseDuration(
            "center") - traci.trafficlight.getNextSwitch("center"))
        self.normCurrPhaseTime = self.currPhaseTime / traci.trafficlight.getPhaseDuration("center")
        self.amberPhase = 1 if current_phase == 1 or current_phase == 3 else 0
        self.currDayTime = (traci.simulation.getTime() / 3600 + self.hourOfTheDay) / 24

    def update_reward(self, veh_ids):
        """
        Updates the reward.
        :param veh_ids: the IDs of the vehicles currently simulated
        :return: None
        """
        rewards = []
        for i in range(len(self.laneIDs)):
            v_max_lane = traci.lane.getMaxSpeed(self.laneIDs[i])
            for x in veh_ids[i]:
                v_max = min(v_max_lane, traci.vehicle.getMaxSpeed(x))
                rewards.append((traci.vehicle.getSpeed(x) - v_max) / v_max)
        self.reward = statistics.mean(rewards) if rewards else 0

    @staticmethod
    def count_detected_veh(ids):
        """
        Counts the number of detected cars in each lane.
        :param ids: the IDs of the vehicles currently simulated
        :return: the number of detected cars in each lane
        """
        cnt = [0] * len(ids)
        for i in range(len(ids)):
            for x in ids[i]:
                if traci.vehicle.getColor(x) == (0, 255, 0, 255):
                    cnt[i] -= 1
        return cnt

    def get_distances(self, veh_ids):
        """
        Computes the distance to the nearest detected vehicle in each lane.
        :param veh_ids: the IDs of the vehicles currently simulated
        :return: the distance to the nearest detected vehicle in each lane
        """
        distances = self.defaultDistances.copy()
        for i in range(len(veh_ids)):
            detected_positions = [traci.vehicle.getLanePosition(x) for x in veh_ids[i] if
                                  traci.vehicle.getColor(x) == (0, 255, 0, 255)]
            if detected_positions:
                distances[i] = distances[i] + max(detected_positions)
        return distances

    def get_state(self):
        """
        Gets the current state representation.
        :return: the current state representation
        """
        return self.detectedCarCnt + self.distanceNearestDetectedVeh + [self.normCurrPhaseTime, self.amberPhase,
                                                                        self.currDayTime]

    def get_reward(self):
        """
        Gets the current reward.
        :return: the current reward
        """
        return self.reward

    def get_episode_end(self):
        """
        Informs if we are at the end of an episode or not.
        :return: 1 if end of episode, 0 otherwise
        """
        return self.episodeEnd

    def get_curr_nb_iterations(self):
        """
        Gets the number of iterations/steps since the beginning of the simulation.
        :return: the current number of iterations
        """
        return self.currNbIterations

    def increment_waiting_time(self, ids):
        """
        Updates the total cumulative waiting time since the beginning of the current episode.
        :param ids: the IDs of the vehicles currently simulated
        :return: None
        """
        cnt = 0
        det_cnt = 0
        undet_cnt = 0
        for i in range(len(ids)):
            for x in ids[i]:
                if traci.vehicle.getSpeed(x) < 0.1:
                    cnt += 1
                    if traci.vehicle.getColor(x) == (0, 255, 0, 255):
                        det_cnt += 1
                    else:
                        undet_cnt += 1
        self.cumWaitingTime += cnt
        self.cumWaitingTimeDetected += det_cnt
        self.cumWaitingTimeUndetected += undet_cnt

    @staticmethod
    def close_simulation():
        """
        Closes the SUMO simulation.
        :return: None
        """
        traci.close()

    def delete_sim_files(self):
        """
        Deletes the SUMO simulation files.
        :return: None
        """
        os.remove("sumo_sim/simple_intersection_" + self.job_id + ".rou.xml")
        os.remove("sumo_sim/simple_intersection_" + self.job_id + ".sumocfg")
