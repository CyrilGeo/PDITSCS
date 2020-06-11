# PDITSCS
This work corresponds to my master thesis about Intelligent Traffic Signal Control using Reinforcement Learning with Partial Detection.

Using PyTorch and SUMO (Simulation of Urban MObility) with the TraCI API.

Nowadays, Artificial Intelligence (AI), through Intelligent Transport Systems (ITS), is seen as a major solution to traffic management problems. Among these issues is the optimization of traffic signal control to reduce traffic congestion. The maturity of connected devices is an opportunity to design cheap and efficient traffic light control systems through Vehicle to Infrastructure (V2I) communications.

While traditional ITS use fixed sensors such as cameras or loop detectors, able to detect every vehicle, a V2I implementation faces a real issue: partial detection. Indeed, it cannot be assumed that every vehicle is equipped with the V2I communication technology required to be observable to the ITS.

The present work reports a V2I Intelligent Transport System meant to reduce congestion at traffic light intersections. Another main goal is to decrease the waiting time of each and every road-user at these intersections. This is why the integration into the algorithm of pedestrians and of a priority system for public transports is also analyzed in this work. The system is implemented with a Reinforcement Learning (RL) algorithm, particularly suitable for this kind of problem since it can use varied inputs and does not have to model the underlying dynamics of the environment as it only relies on experience about the efficiency of its actions.

The analyses under different car flows, detection rates and topologies show that the presented algorithm is able to efficiently reduce commute time of road-users compared to currently deployed fixed time systems, even at low detection rates. It is also robust enough to be implemented on real traffic light intersections. Analyses on road networks with several intersections have shown that the algorithm is able to provide good performances, even without an explicit multi-agent strategy. The integration of a priority system for public transports was successful as well, although an effective integration of pedestrians would require a more complex solution than the one proposed in this work.
