"""
Simulator of the deployment intersection for the LuST deployment case.
"""

import os
import sys
import optparse
import statistics
import random

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


class LuxSim:
    """
    The simulator.
    """

    def __init__(self, nb_episodes, detection_rate, min_phase_duration, gui=False):
        self.N = nb_episodes
        self.episodeCnt = 1
        self.detectionRate = detection_rate
        self.minPhaseDuration = min_phase_duration
        self.hourOfTheDay = 0
        self.currNbIterations = 0
        self.currNbSteps = 0
        self.currPhaseTime = None
        self.laneIDs = ["--31272#7_0", "--31272#7_1", "--31272#7_2",
                        "-30892#16_0", "-30892#16_1", "-30892#16_2",
                        "-31272#6_0", "-31272#6_1", "-31272#6_2",
                        "--30892#17_0", "--30892#17_1", "--30892#17_2"]
        self.detectionDict = {}
        self.sumoBinary = None
        self.episodeEnd = 1  # 1 if last step of an episode, 0 otherwise

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
        self.nbGeneratedVeh = 0
        self.averageWaitingTimes = []
        self.hourlyReward = []
        self.averageHourlyReward = []
        self.averageHourlyRewards = [[]] * 24
        self.hourlyCumWaitingTime = 0
        self.hourlyNbGeneratedVeh = 0
        self.averageHourlyWaitingTime = []
        self.averageHourlyWaitingTimes = [[]] * 24

        # Determines whether to use the simulator's GUI or not
        options = get_options()
        if gui:
            if options.nogui:
                self.sumoBinary = checkBinary("sumo")
            else:
                self.sumoBinary = checkBinary("sumo-gui")
        else:
            self.sumoBinary = checkBinary("sumo")

        self.init_new_episode()

        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.maxLaneDist = min([traci.lane.getLength(x) for x in self.laneIDs])
        self.defaultDistances = [-self.maxLaneDist] * len(veh_ids)
        self.add_veh_to_detection_dict(veh_ids)
        self.update_state(veh_ids)

    def init_new_episode(self):
        """
        Initializes a new episode.
        :return: None
        """
        print("LOADING NEW EPISODE")
        # Starting sumo as a subprocess
        traci.start([self.sumoBinary, "-c", "../LuSTScenario/scenario/dua.static.sumocfg"])

    def add_veh_to_detection_dict(self, veh_ids):
        """
        Updates a vehicle dictionary with the vehicles currently simulated.
        :param veh_ids: the IDs of the vehicles currently simulated
        :return: None
        """
        for i in range(len(self.laneIDs)):
            for x in veh_ids[i]:
                if traci.lane.getLength(self.laneIDs[i]) - traci.vehicle.getLanePosition(
                        x) <= self.maxLaneDist and x not in self.detectionDict:
                    self.nbGeneratedVeh += 1
                    self.hourlyNbGeneratedVeh += 1
                    if random.uniform(0, 1) < self.detectionRate:
                        self.detectionDict[x] = True
                    else:
                        self.detectionDict[x] = False

    def step(self, action=None):
        """
        Performs one iteration/step in the simulator (environment).
        :param action: the action to perform
        :return: True if simulation is to be continued, False otherwise
        """
        self.episodeEnd = 0

        # Episode stops when all raw files have been exhausted (no vehicles left in the simulation)
        if traci.simulation.getMinExpectedNumber() <= 0:
            print("END OF EPISODE")
            traci.close()
            print("EPISODE", self.episodeCnt, "DONE")
            average_reward = statistics.mean(self.rewards)
            print("Average reward:", average_reward)
            self.episodes.append(self.episodeCnt)
            self.averageRewards.append(average_reward)
            average_waiting_time = self.cumWaitingTime / self.nbGeneratedVeh if self.nbGeneratedVeh != 0 else 0
            print("Average waiting time:", average_waiting_time)
            self.averageWaitingTimes.append(average_waiting_time)
            if self.currNbSteps < 86400:
                self.averageHourlyReward.append(
                    statistics.mean(self.hourlyReward + [0] * (3600 - self.currNbSteps % 3600)))
                index = int(self.currNbSteps / 3600)
                self.averageHourlyReward = self.averageHourlyReward + [0] * (23 - index)
                self.averageHourlyWaitingTime.append(self.hourlyCumWaitingTime / self.hourlyNbGeneratedVeh)
                self.averageHourlyWaitingTime = self.averageHourlyWaitingTime + [0] * (23 - index)
            for i in range(len(self.averageHourlyRewards)):
                self.averageHourlyRewards[i] = self.averageHourlyRewards[i] + [self.averageHourlyReward[i]]
            for i in range(len(self.averageHourlyWaitingTimes)):
                self.averageHourlyWaitingTimes[i] = self.averageHourlyWaitingTimes[i] + [
                    self.averageHourlyWaitingTime[i]]
            self.rewards.clear()
            self.averageHourlyReward.clear()
            self.averageHourlyWaitingTime.clear()
            self.cumWaitingTime = 0
            self.nbGeneratedVeh = 0
            self.hourlyReward.clear()
            self.hourlyCumWaitingTime = 0
            self.hourlyNbGeneratedVeh = 0
            self.currNbSteps = 0
            self.detectionDict.clear()

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
        if action is not None and action == 1 and self.currPhaseTime >= self.minPhaseDuration:
            self.next_phase()

        traci.simulationStep()
        self.currNbIterations += 1
        self.currNbSteps += 1
        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.add_veh_to_detection_dict(veh_ids)
        self.update_state(veh_ids)
        self.update_reward(veh_ids)
        self.increment_waiting_time(veh_ids)
        self.rewards.append(self.reward)
        self.hourlyReward.append(self.reward)
        if 86400 >= self.currNbSteps > 0 == self.currNbSteps % 3600:
            self.averageHourlyReward.append(statistics.mean(self.hourlyReward))
            self.averageHourlyWaitingTime.append(
                self.hourlyCumWaitingTime / self.hourlyNbGeneratedVeh if self.hourlyNbGeneratedVeh != 0 else 0)
            self.hourlyReward.clear()
            self.hourlyCumWaitingTime = 0
            self.hourlyNbGeneratedVeh = 0
        return True

    @staticmethod
    def next_phase():
        """
        Switches traffic lights to the next phase.
        :return: None
        """
        traci.trafficlight.setPhase("-12408", (traci.trafficlight.getPhase("-12408") + 1) % 8)

    def update_state(self, veh_ids):
        """
        Updates the state representation.
        :param veh_ids: the IDs of the vehicles currently simulated
        :return: None
        """
        self.detectedCarCnt = self.count_detected_veh(veh_ids)
        self.distanceNearestDetectedVeh = [-x / y for x, y in zip(self.get_distances(veh_ids), self.defaultDistances)]
        current_phase = traci.trafficlight.getPhase("-12408")
        if current_phase == 0:
            self.detectedCarCnt[0] = -self.detectedCarCnt[0]
            self.detectedCarCnt[1] = -self.detectedCarCnt[1]
            self.detectedCarCnt[2] = -self.detectedCarCnt[2]
            self.detectedCarCnt[6] = -self.detectedCarCnt[6]
            self.detectedCarCnt[7] = -self.detectedCarCnt[7]
            self.detectedCarCnt[8] = -self.detectedCarCnt[8]
            self.distanceNearestDetectedVeh[0] = -self.distanceNearestDetectedVeh[0]
            self.distanceNearestDetectedVeh[1] = -self.distanceNearestDetectedVeh[1]
            self.distanceNearestDetectedVeh[2] = -self.distanceNearestDetectedVeh[2]
            self.distanceNearestDetectedVeh[6] = -self.distanceNearestDetectedVeh[6]
            self.distanceNearestDetectedVeh[7] = -self.distanceNearestDetectedVeh[7]
            self.distanceNearestDetectedVeh[8] = -self.distanceNearestDetectedVeh[8]
        elif current_phase == 1 or current_phase == 2:
            self.detectedCarCnt[2] = -self.detectedCarCnt[2]
            self.detectedCarCnt[8] = -self.detectedCarCnt[8]
            self.distanceNearestDetectedVeh[2] = -self.distanceNearestDetectedVeh[2]
            self.distanceNearestDetectedVeh[8] = -self.distanceNearestDetectedVeh[8]
        elif current_phase == 4:
            self.detectedCarCnt[0] = -self.detectedCarCnt[0]
            self.detectedCarCnt[3] = -self.detectedCarCnt[3]
            self.detectedCarCnt[4] = -self.detectedCarCnt[4]
            self.detectedCarCnt[5] = -self.detectedCarCnt[5]
            self.detectedCarCnt[6] = -self.detectedCarCnt[6]
            self.detectedCarCnt[9] = -self.detectedCarCnt[9]
            self.detectedCarCnt[10] = -self.detectedCarCnt[10]
            self.detectedCarCnt[11] = -self.detectedCarCnt[11]
            self.distanceNearestDetectedVeh[0] = -self.distanceNearestDetectedVeh[0]
            self.distanceNearestDetectedVeh[3] = -self.distanceNearestDetectedVeh[3]
            self.distanceNearestDetectedVeh[4] = -self.distanceNearestDetectedVeh[4]
            self.distanceNearestDetectedVeh[5] = -self.distanceNearestDetectedVeh[5]
            self.distanceNearestDetectedVeh[6] = -self.distanceNearestDetectedVeh[6]
            self.distanceNearestDetectedVeh[9] = -self.distanceNearestDetectedVeh[9]
            self.distanceNearestDetectedVeh[10] = -self.distanceNearestDetectedVeh[10]
            self.distanceNearestDetectedVeh[11] = -self.distanceNearestDetectedVeh[11]
        elif current_phase == 5 or current_phase == 6:
            self.detectedCarCnt[5] = -self.detectedCarCnt[5]
            self.detectedCarCnt[11] = -self.detectedCarCnt[11]
            self.distanceNearestDetectedVeh[5] = -self.distanceNearestDetectedVeh[5]
            self.distanceNearestDetectedVeh[11] = -self.distanceNearestDetectedVeh[11]
        self.currPhaseTime = (traci.simulation.getTime() + traci.trafficlight.getPhaseDuration(
            "-12408") - traci.trafficlight.getNextSwitch("-12408"))
        self.normCurrPhaseTime = self.currPhaseTime / traci.trafficlight.getPhaseDuration("-12408")
        self.amberPhase = 1 if current_phase == 1 or current_phase == 3 or current_phase == 5 or current_phase == 7 \
            else 0
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
                if traci.lane.getLength(self.laneIDs[i]) - traci.vehicle.getLanePosition(x) <= self.maxLaneDist:
                    v_max = min(v_max_lane, traci.vehicle.getMaxSpeed(x))
                    rewards.append((traci.vehicle.getSpeed(x) - v_max) / v_max)
        self.reward = statistics.mean(rewards) if rewards else 0

    def count_detected_veh(self, ids):
        """
        Counts the number of detected cars in each lane.
        :param ids: the IDs of the vehicles currently simulated
        :return: the number of detected cars in each lane
        """
        cnt = [0] * len(ids)
        for i in range(len(ids)):
            for x in ids[i]:
                if traci.lane.getLength(self.laneIDs[i]) - traci.vehicle.getLanePosition(x) <= self.maxLaneDist and \
                        self.detectionDict[x]:
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
                                  traci.lane.getLength(self.laneIDs[i]) - traci.vehicle.getLanePosition(
                                      x) <= self.maxLaneDist and self.detectionDict[x]]
            if detected_positions:
                distances[i] = max(detected_positions) - traci.lane.getLength(self.laneIDs[i])
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

    def increment_waiting_time(self, veh_ids):
        """
        Updates the total cumulative waiting time since the beginning of the current episode.
        :param veh_ids: the IDs of the vehicles currently simulated
        :return: None
        """
        cnt = 0
        for i in range(len(veh_ids)):
            for x in veh_ids[i]:
                if traci.vehicle.getSpeed(x) < 0.1 and traci.lane.getLength(
                        self.laneIDs[i]) - traci.vehicle.getLanePosition(x) <= self.maxLaneDist:
                    cnt += 1
        self.cumWaitingTime += cnt
        self.hourlyCumWaitingTime += cnt

    @staticmethod
    def close_simulation():
        """
        Closes the SUMO simulation.
        :return: None
        """
        traci.close()
