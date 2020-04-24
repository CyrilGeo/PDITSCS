import os
import sys
import optparse
import random
import statistics
import pickle
import matplotlib.pyplot as plt

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
        self.nbGeneratedVeh = 0
        self.averageWaitingTimes = []

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
        <net-file value="simple_intersection_phase50.net.xml"/>
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

    # Initializes a new episode
    def init_new_episode(self):
        print("LOADING NEW EPISODE")
        self.generate_traffic(self.hourlyProbs)

        # Starting sumo as a subprocess
        traci.start([self.sumoBinary, "-c", "sumo_sim/simple_intersection_" + self.job_id + ".sumocfg"])
        '''traci.start(
            [self.sumoBinary, "-c", "sumo_sim/simple_intersection_" + self.job_id + ".sumocfg", "--tripinfo-output",
             "tripinfo_" + self.job_id + ".xml"])'''

    # Randomly generates the route file that determines the traffic in the simulation
    def generate_traffic(self, hourly_probs=None):
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
                            print('    <vehicle id="' + str(self.nbGeneratedVeh) + '" type="car" route="route' + str(
                                j) + '" depart="' + str(
                                i) + '" color="' + self.select_color() + '" departSpeed="max"/>', file=routes)
                            self.nbGeneratedVeh += 1
            else:
                for i in range(self.n):
                    for j in range(len(self.routeProbs)):
                        if random.uniform(0, 1) < self.routeProbs[j]:
                            print('    <vehicle id="' + str(self.nbGeneratedVeh) + '" type="car" route="route' + str(
                                j) + '" depart="' + str(
                                i) + '" color="' + self.select_color() + '" departSpeed="max"/>', file=routes)
                            self.nbGeneratedVeh += 1

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
            self.rewards.clear()
            self.cumWaitingTime = 0
            self.nbGeneratedVeh = 0

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

        # Randomly choosing if the simulation switches to the next state or stays at the current state
        '''if random.uniform(0, 1) < 0.02 and self.currPhaseTime >= self.minPhaseDuration:
            self.next_phase()'''

        traci.simulationStep()
        self.currNbIterations += 1
        veh_ids = [traci.lane.getLastStepVehicleIDs(x) for x in self.laneIDs]
        self.update_state(veh_ids)
        self.update_reward(veh_ids)
        self.increment_waiting_time(veh_ids)
        self.rewards.append(self.reward)
        return True

    # Switches traffic light to the next phase
    @staticmethod
    def next_phase():
        traci.trafficlight.setPhase("center", (traci.trafficlight.getPhase("center") + 1) % 4)

    # Updates the state values
    def update_state(self, veh_ids):
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

    def increment_waiting_time(self, ids):
        cnt = 0
        for i in range(len(ids)):
            for x in ids[i]:
                if traci.vehicle.getSpeed(x) < 0.1:
                    cnt += 1
        self.cumWaitingTime += cnt

    # /!\ Filename without file extension
    def save_stats(self, gen_name):
        # For loading, use "rb" and pickle.load(file)
        with open(os.path.join("data", gen_name + "_episodes.txt"), "wb") as file:
            pickle.dump(self.episodes, file)
        with open(os.path.join("data", gen_name + "_rewards.txt"), "wb") as file:
            pickle.dump(self.averageRewards, file)
        with open(os.path.join("data", gen_name + "_waiting_times.txt"), "wb") as file:
            pickle.dump(self.averageWaitingTimes, file)

        plt.figure()
        plt.plot(self.episodes, self.averageRewards, color="steelblue")
        plt.xlabel("Episode")
        plt.ylabel("Average reward")
        plt.savefig(os.path.join("figures", "previews", "out_" + gen_name + "_r.png"))

        plt.figure()
        plt.plot(self.episodes, self.averageWaitingTimes, color="steelblue")
        plt.xlabel("Episode")
        plt.ylabel("Average waiting time (s)")
        plt.savefig(os.path.join("figures", "previews", "out_" + gen_name + "_w.png"))

    @staticmethod
    def close_simulation():
        traci.close()

    def delete_sim_files(self):
        os.remove("sumo_sim/simple_intersection_" + self.job_id + ".rou.xml")
        os.remove("sumo_sim/simple_intersection_" + self.job_id + ".sumocfg")
