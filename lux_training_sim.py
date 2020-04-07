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


class LuxTrainingSim:

    def __init__(self, nb_episodes, detection_rate, min_phase_duration, gui=False):
        self.N = nb_episodes
        self.episodeCnt = 1
        self.detectionRate = detection_rate
        self.minPhaseDuration = min_phase_duration
        self.hourOfTheDay = 0
        self.currNbIterations = 0
        self.currNbSteps = 0
        self.currPhaseTime = None
        self.detectedColor = "0, 255, 0"
        self.undetectedColor = "255, 0, 0"
        self.laneIDs = ["in_west_0", "in_west_1", "in_west_2",
                        "in_north_0", "in_north_1", "in_north_2",
                        "in_east_0", "in_east_1", "in_east_2",
                        "in_south_0", "in_south_1", "in_south_2"]
        self.detectionDict = {}
        self.sumoBinary = None
        self.episodeEnd = 1  # 1 if last step of an episode, 0 otherwise
        self.hourlyProbs = [[0.0003] * 3 + [0.0011] * 3 + [0.0015] * 3 + [0.0029] * 3,
                            [0.0005] * 3 + [0.002] * 3 + [0.0016] * 3 + [0.002] * 3,
                            [0.0005] * 3 + [0.0019] * 3 + [0.0018] * 3 + [0.0027] * 3,
                            [0.0003] * 3 + [0.0024] * 3 + [0.0016] * 3 + [0.0023] * 3,
                            [0.0035] * 3 + [0.0032] * 3 + [0.004] * 3 + [0.0061] * 3,
                            [0.0197] * 3 + [0.012] * 3 + [0.0094] * 3 + [0.0186] * 3,
                            [0.05] * 3 + [0.0296] * 3 + [0.0229] * 3 + [0.0481] * 3,
                            [0.0717] * 3 + [0.0507] * 3 + [0.0375] * 3 + [0.0653] * 3,
                            [0.0756] * 3 + [0.0529] * 3 + [0.0434] * 3 + [0.0742] * 3,
                            [0.0696] * 3 + [0.0393] * 3 + [0.0331] * 3 + [0.0603] * 3,
                            [0.0541] * 3 + [0.025] * 3 + [0.0212] * 3 + [0.0406] * 3,
                            [0.0284] * 3 + [0.0208] * 3 + [0.0184] * 3 + [0.0293] * 3,
                            [0.0419] * 3 + [0.0331] * 3 + [0.0294] * 3 + [0.0540] * 3,
                            [0.0569] * 3 + [0.0373] * 3 + [0.0327] * 3 + [0.0522] * 3,
                            [0.0465] * 3 + [0.0263] * 3 + [0.0244] * 3 + [0.0542] * 3,
                            [0.0157] * 3 + [0.0198] * 3 + [0.0214] * 3 + [0.0313] * 3,
                            [0.0132] * 3 + [0.0426] * 3 + [0.0374] * 3 + [0.0517] * 3,
                            [0.0156] * 3 + [0.0594] * 3 + [0.0575] * 3 + [0.0702] * 3,
                            [0.0234] * 3 + [0.0616] * 3 + [0.0725] * 3 + [0.0712] * 3,
                            [0.0163] * 3 + [0.0556] * 3 + [0.0572] * 3 + [0.0668] * 3,
                            [0.0087] * 3 + [0.0363] * 3 + [0.0328] * 3 + [0.0387] * 3,
                            [0.0063] * 3 + [0.0224] * 3 + [0.0259] * 3 + [0.0271] * 3,
                            [0.0044] * 3 + [0.0183] * 3 + [0.0165] * 3 + [0.0274] * 3,
                            [0.0037] * 3 + [0.0171] * 3 + [0.0196] * 3 + [0.0256] * 3]
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
        self.nbGeneratedVeh = 0
        self.averageWaitingTimes = []
        self.hourlyReward = []
        self.averageHourlyReward = []
        self.averageHourlyRewards = [[]] * 24
        self.hourlyCumWaitingTime = 0
        self.hourlyNbGeneratedVeh = []
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

        with open("sumo_sim/LuST_training_intersection_" + self.job_id + ".sumocfg", "w") as config:
            print("""<configuration>
    <input>
        <net-file value="LuST_training_intersection.net.xml"/>
        <route-files value="LuST_training_intersection_""" + self.job_id + """.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="86400"/>
    </time>
</configuration>""", file=config)

        self.init_new_episode()

        self.defaultDistances = [-traci.lane.getLength(x) for x in self.laneIDs]
        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.update_state(veh_ids)

    # Initializes a new episode
    def init_new_episode(self):
        print("LOADING NEW EPISODE")
        self.generate_traffic()

        # Starting sumo as a subprocess
        traci.start([self.sumoBinary, "-c", "sumo_sim/LuST_training_intersection_" + self.job_id + ".sumocfg"])

    # Randomly generates the route file that determines the traffic in the simulation
    def generate_traffic(self):
        random.seed()

        with open("sumo_sim/LuST_training_intersection_" + self.job_id + ".rou.xml", "w") as routes:
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

            nb_configurations = len(self.hourlyProbs)
            k = -1
            cnt_hourly_veh = 0
            for i in range(86400):
                if i % 3600 == 0:
                    self.hourlyNbGeneratedVeh.append(cnt_hourly_veh)
                    cnt_hourly_veh = 0
                    k = (k + 1) % nb_configurations
                for j in range(len(self.hourlyProbs[k])):
                    if random.uniform(0, 1) < self.hourlyProbs[k][j]:
                        print('    <vehicle id="' + str(self.nbGeneratedVeh) + '" type="car" route="route' + str(
                            j) + '" depart="' + str(i) + '" color="' + self.select_color() + '" departSpeed="max"/>',
                              file=routes)
                        self.nbGeneratedVeh += 1
                        cnt_hourly_veh += 1

            print("</routes>", file=routes)

    # Randomly chooses if a generated vehicle is detected or not by selecting its color accordingly
    def select_color(self):
        if random.uniform(0, 1) < self.detectionRate:
            return self.detectedColor
        return self.undetectedColor

    # Performs one iteration/step in the simulator (environment) and returns the new state and the reward
    # use step(self) for use in testing.py
    def step(self, action=None):
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
            print("Average waiting time:", average_waiting_time)
            self.averageWaitingTimes.append(average_waiting_time)
            if self.currNbSteps < 86400:
                self.averageHourlyReward.append(
                    statistics.mean(self.hourlyReward + [0] * (3600 - self.currNbSteps % 3600)))
                index = int(self.currNbSteps / 3600)
                self.averageHourlyReward = self.averageHourlyReward + [0] * (23 - index)
                self.averageHourlyWaitingTime.append(self.hourlyCumWaitingTime / self.hourlyNbGeneratedVeh[index])
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
            self.hourlyNbGeneratedVeh.clear()
            self.currNbSteps = 0

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
        if action is not None and action == 1 and self.currPhaseTime > self.minPhaseDuration:
            self.next_phase()

        traci.simulationStep()
        self.currNbIterations += 1
        self.currNbSteps += 1
        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.update_state(veh_ids)
        self.update_reward(veh_ids)
        self.increment_waiting_time(veh_ids)
        self.rewards.append(self.reward)
        self.hourlyReward.append(self.reward)
        if 86400 >= self.currNbSteps > 0 == self.currNbSteps % 3600:
            self.averageHourlyReward.append(statistics.mean(self.hourlyReward))
            index = int(self.currNbSteps / 3600) - 1
            self.averageHourlyWaitingTime.append(
                self.hourlyCumWaitingTime / self.hourlyNbGeneratedVeh[index] if self.hourlyNbGeneratedVeh[
                                                                                    index] != 0 else 0)
            self.hourlyReward.clear()
            self.hourlyCumWaitingTime = 0
        return True

    # Switches traffic light to the next phase
    @staticmethod
    def next_phase():
        traci.trafficlight.setPhase("center", (traci.trafficlight.getPhase("center") + 1) % 8)

    # Updates the state values
    def update_state(self, veh_ids):
        self.detectedCarCnt = self.count_detected_veh(veh_ids)
        self.distanceNearestDetectedVeh = [-x / y for x, y in zip(self.get_distances(veh_ids), self.defaultDistances)]
        current_phase = traci.trafficlight.getPhase("center")
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
            "center") - traci.trafficlight.getNextSwitch("center"))
        self.normCurrPhaseTime = self.currPhaseTime / traci.trafficlight.getPhaseDuration("center")
        self.amberPhase = 1 if current_phase == 1 or current_phase == 3 or current_phase == 5 or current_phase == 7 \
            else 0
        self.currDayTime = (traci.simulation.getTime() / 3600 + self.hourOfTheDay) / 24

    # Updates the reward
    def update_reward(self, veh_ids):
        rewards = []
        for i in range(len(self.laneIDs)):
            v_max_lane = traci.lane.getMaxSpeed(self.laneIDs[i])
            for x in veh_ids[i]:
                v_max = min(v_max_lane, traci.vehicle.getMaxSpeed(x))
                rewards.append((traci.vehicle.getSpeed(x) - v_max) / v_max)
        self.reward = statistics.mean(rewards) if rewards else 0

    # Returns the number of detected cars in each lane, given a list of ids of all cars for each lane
    @staticmethod
    def count_detected_veh(ids):
        cnt = [0] * len(ids)
        for i in range(len(ids)):
            for x in ids[i]:
                if traci.vehicle.getColor(x) == (0, 255, 0, 255):
                    cnt[i] -= 1
        return cnt

    # Returns the distance to the nearest detected vehicle in each lane, given lane ids and their corresponding list of
    # car ids
    def get_distances(self, veh_ids):
        distances = self.defaultDistances.copy()
        for i in range(len(veh_ids)):
            detected_positions = [traci.vehicle.getLanePosition(x) for x in veh_ids[i] if
                                  traci.vehicle.getColor(x) == (0, 255, 0, 255)]
            if detected_positions:
                distances[i] = distances[i] + max(detected_positions)
        return distances

    # Returns the state values as a list
    def get_state(self):
        return self.detectedCarCnt + self.distanceNearestDetectedVeh + [self.normCurrPhaseTime, self.amberPhase,
                                                                        self.currDayTime]

    # Returns the reward
    def get_reward(self):
        return self.reward

    # Returns 1 if end of episode, 0 otherwise
    def get_episode_end(self):
        return self.episodeEnd

    def get_curr_nb_iterations(self):
        return self.currNbIterations

    def increment_waiting_time(self, veh_ids):
        cnt = 0
        for i in range(len(veh_ids)):
            for x in veh_ids[i]:
                if traci.vehicle.getSpeed(x) < 0.1:
                    cnt += 1
        self.cumWaitingTime += cnt
        self.hourlyCumWaitingTime += cnt

    @staticmethod
    def close_simulation():
        traci.close()

    def delete_sim_files(self):
        os.remove("sumo_sim/LuST_training_intersection_" + self.job_id + ".rou.xml")
        os.remove("sumo_sim/LuST_training_intersection_" + self.job_id + ".sumocfg")
