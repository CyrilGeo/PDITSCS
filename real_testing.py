import lux_sim as sim

if __name__ == "__main__":
    simulator = sim.LuxSim(True)
    simulator.start_sim()
    while simulator.step():
        pass
    simulator.close()
