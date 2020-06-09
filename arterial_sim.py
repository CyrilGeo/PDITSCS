"""
Simulator of the 5x1 arterial road.
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


class ArterialSimulator:
    """
    The simulator.
    """

    def __init__(self, nb_episodes, nb_episode_steps, detection_rate, min_phase_duration, route_probs, hour_of_the_day,
                 gui=False):
        self.N = nb_episodes
        self.n = nb_episode_steps
        self.episodeCnt = 1
        self.detectionRate = detection_rate
        self.minPhaseDuration = min_phase_duration
        self.hourOfTheDay = hour_of_the_day
        self.currNbIterations = 0
        self.currPhaseTime = []
        self.routeProbs = route_probs
        self.detectedColor = "0, 255, 0"
        self.undetectedColor = "255, 0, 0"
        self.intersections = [3, 6, 9, 12, 15]
        # Computes the lane IDs
        self.laneIDs = []
        for inter in self.intersections:
            lane_ids = [x + str(inter) + "_0" for x in [str(inter - 3), str(inter + 1), str(inter + 3), str(inter + 2)]]
            self.laneIDs.append(lane_ids)
        # Computes the possible routes in the road network
        self.routes = []
        outside_nodes = [0, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18]
        for i in outside_nodes:
            for j in outside_nodes:
                if i != j:
                    route = ""
                    k = i
                    if i % 3 != 0:
                        route += str(i) + str(i - i % 3) + " "
                        k = i - i % 3
                    if i < j:
                        while k + 3 <= j:
                            route += str(k) + str(k + 3) + " "
                            k += 3
                        if j % 3 != 0:
                            route += str(k) + str(j)
                    else:
                        while k - 3 >= j - 3 and k != 0:
                            route += str(k) + str(k - 3) + " "
                            k -= 3
                        if j % 3 != 0:
                            route += str(k) + str(j)
                    self.routes.append(route)
        self.sumoBinary = None
        self.episodeEnd = 1  # 1 if last step of an episode, 0 otherwise
        self.job_id = "0"
        if "SLURM_JOB_ID" in os.environ:
            self.job_id = os.environ["SLURM_JOB_ID"]

        # State variables
        self.detectedCarCnt = []
        self.distanceNearestDetectedVeh = []
        self.normCurrPhaseTime = []
        self.amberPhase = []
        self.currDayTime = None

        # Stats
        self.episodes = []
        self.reward = []
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

        with open("sumo_sim/arterial_" + self.job_id + ".sumocfg", "w") as config:
            print("""<configuration>
    <input>
        <net-file value="arterial.net.xml"/>
        <route-files value="arterial_""" + self.job_id + """.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value=""" + '"' + str(self.n) + '"' + """/>
    </time>
</configuration>""", file=config)

        self.init_new_episode()

        self.defaultDistances = [[-traci.lane.getLength(x) for x in lane_ids] for lane_ids in self.laneIDs]
        veh_ids = [[traci.lane.getLastStepVehicleIDs(x) for x in lane_ids] for lane_ids in self.laneIDs]
        for i in range(len(veh_ids)):
            self.currDayTime = (traci.simulation.getTime() / 3600 + self.hourOfTheDay) / 24
            self.update_state(veh_ids[i], str(self.intersections[i]), self.defaultDistances[i])

    def init_new_episode(self):
        """
        Initializes a new episode.
        :return: None
        """
        print("LOADING NEW EPISODE")
        self.generate_traffic()

        # Starting sumo as a subprocess
        traci.start([self.sumoBinary, "-c", "sumo_sim/arterial_" + self.job_id + ".sumocfg"])

    def generate_traffic(self):
        """
        Randomly generates the route file that determines the traffic in the simulation.
        :return: None
        """
        random.seed()

        with open("sumo_sim/arterial_" + self.job_id + ".rou.xml", "w") as routes:
            print("""<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" maxSpeed="55.55" length="4.5"/>""", file=routes)

            for i in range(len(self.routes)):
                print('    <route id="route' + str(i) + '" edges="' + self.routes[i] + '"/>', file=routes)

            # Randomly choosing if a vehicle is generated for each step and each route
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

    def step(self, actions=None):
        """
        Performs one iteration/step in the simulator (environment).
        :param actions: the actions to perform on all the intersections
        :return: True if simulation is to be continued, False otherwise
        """
        self.episodeEnd = 0

        # Episode stops when all raw files have been exhausted (no vehicles left in the simulation)
        if traci.simulation.getMinExpectedNumber() <= 0:
            traci.close()
            print("EPISODE", self.episodeCnt, "DONE")
            average_reward = statistics.mean([statistics.mean(x) for x in self.rewards])
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
        if actions is not None:
            for i in range(len(self.intersections)):
                if actions[i] == 1 and self.currPhaseTime[i] >= self.minPhaseDuration:
                    self.next_phase(str(self.intersections[i]))

        # Fixed phase duration
        else:
            for i in range(len(self.intersections)):
                if (self.currPhaseTime[i] >= 10 and traci.trafficlight.getPhase(str(self.intersections[i])) == 0) or \
                        self.currPhaseTime[i] >= 30 and traci.trafficlight.getPhase(str(self.intersections[i])) == 2:
                    self.next_phase(str(self.intersections[i]))
        '''else:
            for i in range(len(self.intersections)):
                if self.currPhaseTime[i] >= 10:
                    self.next_phase(str(self.intersections[i]))'''

        traci.simulationStep()
        self.currNbIterations += 1
        veh_ids = [[traci.lane.getLastStepVehicleIDs(x) for x in lane_ids] for lane_ids in self.laneIDs]
        self.clear_state()
        self.reward.clear()
        for i in range(len(veh_ids)):
            self.currDayTime = (traci.simulation.getTime() / 3600 + self.hourOfTheDay) / 24
            self.update_state(veh_ids[i], str(self.intersections[i]), self.defaultDistances[i])
            self.update_reward(veh_ids[i], self.laneIDs[i])
        self.increment_waiting_time(veh_ids)
        self.rewards.append(self.reward.copy())
        return True

    @staticmethod
    def next_phase(intersection):
        """
        Switches the traffic lights of a given intersection to the next phase.
        :param intersection: the given intersection
        :return: None
        """
        traci.trafficlight.setPhase(intersection, (traci.trafficlight.getPhase(intersection) + 1) % 4)

    def clear_state(self):
        """
        Erases the state values of the previous state
        :return: None
        """
        self.detectedCarCnt.clear()
        self.distanceNearestDetectedVeh.clear()
        self.currPhaseTime.clear()
        self.normCurrPhaseTime.clear()
        self.amberPhase.clear()

    def update_state(self, veh_ids, intersection, default_distances):
        """
        Updates the state representation of a given intersection.
        :param veh_ids: the IDs of the vehicles currently simulated
        :param intersection: the given intersection
        :param default_distances: the distances of the lanes in the intersection
        :return: None
        """
        self.detectedCarCnt.append(self.count_detected_veh(veh_ids))
        self.distanceNearestDetectedVeh.append(
            [-x / y for x, y in zip(self.get_distances(veh_ids, default_distances), default_distances)])
        current_phase = traci.trafficlight.getPhase(intersection)
        if current_phase == 0:
            self.detectedCarCnt[-1][1] = -self.detectedCarCnt[-1][1]
            self.detectedCarCnt[-1][3] = -self.detectedCarCnt[-1][3]
            self.distanceNearestDetectedVeh[-1][1] = -self.distanceNearestDetectedVeh[-1][1]
            self.distanceNearestDetectedVeh[-1][3] = -self.distanceNearestDetectedVeh[-1][3]
        elif current_phase == 2:
            self.detectedCarCnt[-1][0] = -self.detectedCarCnt[-1][0]
            self.detectedCarCnt[-1][2] = -self.detectedCarCnt[-1][2]
            self.distanceNearestDetectedVeh[-1][0] = -self.distanceNearestDetectedVeh[-1][0]
            self.distanceNearestDetectedVeh[-1][2] = -self.distanceNearestDetectedVeh[-1][2]

        self.currPhaseTime.append((traci.simulation.getTime() + traci.trafficlight.getPhaseDuration(
            intersection) - traci.trafficlight.getNextSwitch(intersection)))
        self.normCurrPhaseTime.append(self.currPhaseTime[-1] / traci.trafficlight.getPhaseDuration(intersection))
        if current_phase == 1 or current_phase == 3:
            self.amberPhase.append(1)
        else:
            self.amberPhase.append(0)

    def update_reward(self, veh_ids, lane_ids):
        """
        Updates the reward for given lanes corresponding to an intersection.
        :param veh_ids: the IDs of the vehicles currently simulated
        :param lane_ids: the given lanes
        :return: None
        """
        rewards = []
        for i in range(len(lane_ids)):
            v_max_lane = traci.lane.getMaxSpeed(lane_ids[i])
            for x in veh_ids[i]:
                v_max = min(v_max_lane, traci.vehicle.getMaxSpeed(x))
                rewards.append((traci.vehicle.getSpeed(x) - v_max) / v_max)
        if rewards:
            self.reward.append(statistics.mean(rewards))
        else:
            self.reward.append(0)

    @staticmethod
    def count_detected_veh(ids):
        """
        Counts the number of detected cars in each lane of an intersection.
        :param ids: the IDs of the vehicles currently simulated in the intersection
        :return: the number of detected cars in each lane of the intersection
        """
        cnt = [0] * len(ids)
        for i in range(len(ids)):
            for x in ids[i]:
                if traci.vehicle.getColor(x) == (0, 255, 0, 255):
                    cnt[i] -= 1
        return cnt

    @staticmethod
    def get_distances(veh_ids, default_distances):
        """
        Computes the distance to the nearest detected vehicle in each lane of an intersection.
        :param veh_ids: the IDs of the vehicles currently simulated in the intersection
        :param default_distances: the distances of the lanes in the intersection
        :return: the distance to the nearest detected vehicle in each lane of the intersection
        """
        distances = default_distances.copy()
        for i in range(len(veh_ids)):
            detected_positions = [traci.vehicle.getLanePosition(x) for x in veh_ids[i] if
                                  traci.vehicle.getColor(x) == (0, 255, 0, 255)]
            if detected_positions:
                distances[i] = distances[i] + max(detected_positions)
        return distances

    def get_state(self):
        """
        Gets the current state representation at each intersection.
        :return: the current state representations
        """
        states = []
        for i in range(len(self.detectedCarCnt)):
            states.append(self.detectedCarCnt[i] + self.distanceNearestDetectedVeh[i] + [self.normCurrPhaseTime[i],
                                                                                         self.amberPhase[i],
                                                                                         self.currDayTime])
        return states

    def get_reward(self):
        """
        Gets the current reward for each intersection.
        :return: the current reward for each intersection
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
        for veh_ids in ids:
            for i in range(len(veh_ids)):
                for x in veh_ids[i]:
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
        os.remove("sumo_sim/arterial_" + self.job_id + ".rou.xml")
        os.remove("sumo_sim/arterial_" + self.job_id + ".sumocfg")
