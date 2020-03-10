import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    with open("data/model_100_medium" + "_episodes.txt", "rb") as file:
        episodes = pickle.load(file)
    with open("data/uniform1over45" + "_baseline_r.txt", "rb") as file:
        baseline_r = pickle.load(file)
    with open("data/uniform1over45" + "_baseline_w.txt", "rb") as file:
        baseline_w = pickle.load(file)
    with open("data/model_100_medium" + "_rewards.txt", "rb") as file:
        rewards1 = pickle.load(file)
    with open("data/model_50_medium" + "_rewards.txt", "rb") as file:
        rewards2 = pickle.load(file)
    with open("data/model_20_medium" + "_rewards.txt", "rb") as file:
        rewards3 = pickle.load(file)
    with open("data/model_100_medium" + "_waiting_times.txt", "rb") as file:
        waiting_times1 = pickle.load(file)
    with open("data/model_50_medium" + "_waiting_times.txt", "rb") as file:
        waiting_times2 = pickle.load(file)
    with open("data/model_20_medium" + "_waiting_times.txt", "rb") as file:
        waiting_times3 = pickle.load(file)

    plt.figure()
    plt.plot(episodes, rewards1, color="limegreen", label="100% detection rate")
    plt.plot(episodes, rewards2, color="steelblue", label="50% detection rate")
    plt.plot(episodes, rewards3, color="gold", label="20% detection rate")
    plt.axhline(y=baseline_r, color="r", label="fixed time (10s)")
    # plt.ylim(bottom=8.0)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("figures/medium_uni_r.png")
    plt.show()

    plt.figure()
    plt.plot(episodes, waiting_times1, color="limegreen", label="100% detection rate")
    plt.plot(episodes, waiting_times2, color="steelblue", label="50% detection rate")
    plt.plot(episodes, waiting_times3, color="gold", label="20% detection rate")
    plt.axhline(y=baseline_w, color="r", label="fixed time (10s)")
    # plt.ylim(top=8.0)
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time (s)")
    plt.legend()
    plt.savefig("figures/medium_uni_w.png")
    plt.show()
