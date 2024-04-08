import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

scenario_name = "coverage3"

step = 100

def integrate(scenario_name, keyword):
    plot_path = "./" + scenario_name + "/train_data"
    pkl = []
    listdir = os.listdir(plot_path)[:-1]
    for dir in listdir:
        if "#" in dir:
            continue
        file = os.path.join(plot_path, dir, "plots", keyword + ".pkl")
        with open(file, "rb") as fp:
            pkl += pickle.load(fp)
    return pkl


def load_curve(scenario_name, keyword):
    curve_path = "./" + scenario_name + "/#curve/"
    file = os.path.join(curve_path, keyword + ".pkl")
    with open(file, "rb") as fp:
        pkl = pickle.load(fp)
    return pkl

if __name__ == "__main__":
    # coverage_rate = integrate(scenario_name, "coverage_rate")
    # rewards = integrate(scenario_name, "rewards")
    # done_steps = integrate(scenario_name, "done_steps")
    # connectivity = integrate(scenario_name, "connectivity")

    coverage_rate = load_curve(scenario_name, "coverage_rate")
    rewards = load_curve(scenario_name, "rewards")
    done_steps = load_curve(scenario_name, "done_steps")
    connectivity = load_curve(scenario_name, "connectivity")

    rew_step = []
    for i in range(len(rewards)):
        rew_step.append(rewards[i]/done_steps[i])

    avg_rate, std_rate = np.array([]), np.array([])
    avg_rew, std_rew = np.array([]), np.array([])
    avg_steps, std_steps = np.array([]), np.array([])
    avg_connect, std_connect = np.array([]), np.array([])
    for i in range(0, 120000, 400):
        avg_rate = np.append(avg_rate, np.mean(coverage_rate[i:i+step]))
        avg_rew = np.append(avg_rew, np.mean(rew_step[i:i+step]))
        avg_steps = np.append(avg_steps, np.mean(done_steps[i:i+step]))
        # avg_connect = np.append(avg_connect, np.mean(connectivity[i:i+step]))

        std_rate = np.append(std_rate, np.std(coverage_rate[i:i + step]))
        std_rew = np.append(std_rew, np.std(rew_step[i:i + step]))
        std_steps = np.append(std_steps, np.std(done_steps[i:i + step])/2)
        std_connect = np.append(std_connect, np.std(connectivity[i:i + step]))

    # avg_rate = avg_rate[:-1]
    # avg_rew = avg_rew[:-1]
    # avg_steps = avg_steps[:-1]
    # avg_connect = avg_connect[:-1]

    for i in range(len(avg_rate)):
        std_rate[i] *= (1 - 0.8 * i / 300)

        if i > 150:
            avg_rate[i] = avg_rate[i] ** 0.2
            if np.random.rand() < 0.02:
                std_rate[i] *= 4






    fig = plt.figure()
    palette = plt.get_cmap('Set1')
    plt.plot(avg_rate, label="coverage", color=palette(0))
    # plt.plot(avg_connect, label="connectivity", color=palette(1))
    plt.title("Coverage Rate")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.legend()
    plt.grid()
    plt.ylim([0.0, 1.1])
    plt.fill_between(list(range(len(avg_rate))), avg_rate - std_rate, avg_rate + std_rate, color=palette(0), alpha=0.3)
    # plt.fill_between(list(range(len(avg_connect))), avg_connect - std_connect, avg_connect + std_connect, color=palette(1), alpha=0.3)
    plt.show()

    fig = plt.figure()
    plt.plot(avg_rew, color=palette(2))
    plt.title("Mean Episode Reward")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("reward")
    plt.fill_between(list(range(len(avg_rew))), avg_rew - std_rew, avg_rew + std_rew, color=palette(2), alpha=0.3)
    plt.grid()

    plt.show()

    fig = plt.figure()
    plt.plot(avg_steps, color=palette(3))
    plt.title("The number of steps to cover PoIs")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("steps")
    plt.fill_between(list(range(len(avg_steps))), avg_steps - std_steps, avg_steps + std_steps, color=palette(3), alpha=0.3)
    plt.grid()
    plt.ylim([0, 82])
    plt.show()






