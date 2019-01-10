import environments as env
import learning_methods as lm

import numpy as np 
from matplotlib import pyplot as plt

if __name__ == "__main__":
    """1.) Gridworld"""
    final_results = {}
    scene = env.Gridworld()
    trials = 100

    # 1.1) Gridworld w/ tabular Sarsa EPS-Greedy
    init = 0.10
    eps = 1e-3
    alpha = 1e-1
    episodes = 100

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        q = np.ones((scene.num_states, len(scene.actions)))*init
        results[trial, :] = lm.tabularSarsa(eps, alpha, episodes, q, scene)

    final_results["Gridworld Sarsa"] = results

    # 1.2) Gridworld w/ tabular Q-learning EPS-Greedy
    init = 0.10
    eps = 1e-3
    alpha = 1e-1
    episodes = 100

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        q = np.ones((scene.num_states, len(scene.actions)))*init
        results[trial, :] = lm.tabularQlearn(eps, alpha, episodes, q, scene)

    final_results["Gridworld Q-Learn"] = results

    # 1.3) Gridworld w/ tabular Sarsa(lam) EPS-Greedy
    init = 0.10
    lam = 0.8
    eps = 1e-3
    alpha = 1e-1
    episodes = 100

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        q = np.ones((scene.num_states, len(scene.actions)))*init
        results[trial, :] = lm.tabularSarsaLam(lam, eps, alpha, episodes, q, scene)

    final_results["Gridworld Sarsa-Lam"] = results

    # 1.4) Gridworld w/ tabular Q-learning(lam) EPS-Greedy
    init = 0.10
    lam = 0.8
    eps = 1e-3
    alpha = 1e-1
    episodes = 100

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        q = np.ones((scene.num_states, len(scene.actions)))*init
        results[trial, :] = lm.tabularQlearnLam(lam, eps, alpha, episodes, q, scene)

    final_results["Gridworld Q-Learn-Lam"] = results

    # 1.5) Gridworld w/ tabular Actor-Critic

    v_init = 0
    theta_init = 0
    lam = 0.2
    alpha = 0.7
    beta = 0.7
    sigma = 4
    episodes = 100

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        v = np.ones(scene.num_states)*init
        theta = np.ones((scene.num_states, len(scene.actions)))*theta_init
        results[trial, :] = lm.tabularActorCritic(lam, alpha, beta, sigma, 
                                                   episodes, v, theta, scene)

    final_results["Gridworld Actor-Critic"] = results

    colors = ["#FD8C8C", "#4286F4", "#F4BE41", "#A641F4", "#3FB22A"]
    for (key, result), color in zip(final_results.items(), colors):
        _, _, bars = plt.errorbar(
            x=np.arange(episodes),
            y=result.mean(axis=0),
            yerr=result.std(axis=0), 
            color=color,
            ecolor=color, 
            elinewidth=1,
            label=key)

        [bar.set_alpha(0.4) for bar in bars]

    plt.xlabel("Episode")
    plt.ylabel("Discounted Return")
    plt.ylim(-5,5)
    plt.yticks(np.arange(-5,5+1,1))
    plt.title("Gridworld Method Comparisons")
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()

    """2.) Mountain Car"""
    final_results = {}
    scene = env.MountainCar()
    trials = 100

    # 2.1) Mountain Car w/ Fourier FuncApprox Sarsa EPS-Greedy
    init = 1
    eps = 0.001
    alpha = 0.005
    episodes = 100
    order = 5

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        w = np.ones((len(scene.actions), (order+1)**scene.num_state_vars))*init
        results[trial, :] = lm.fncApproxSarsa(eps, alpha, episodes, order, w, scene)

    final_results["Mountain Car Sarsa"] = results

    # 2.2) Mountain Car w/ Fourier FuncApprox Qlearn EPS-Greedy
    init = 1
    eps = 0.001
    alpha = 0.005
    episodes = 100
    order = 5

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        w = np.ones((len(scene.actions), (order+1)**scene.num_state_vars))*init
        results[trial, :] = lm.fncApproxQlearn(eps, alpha, episodes, order, w, scene)

    final_results["Mountain Car Q-Learn"] = results

    # 2.3 Mountain Car w/ Fourier FuncApprox SarsaLam EPS-Greedy
    init = 0
    lam = 0.8
    eps = 0.001
    alpha = 0.001
    episodes = 100
    order = 5

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        w = np.ones((len(scene.actions), (order+1)**scene.num_state_vars))*init
        results[trial, :] = lm.fncApproxSarsaLam(lam, eps, alpha, episodes, order, w, scene)

    final_results["Mountain Car Sarsa-Lam"] = results
    
    # 2.4 Mountain Car w/ Fourier FuncApprox QLearnLam EPS-Greedy
    init = 0
    lam = 0.8
    eps = 0.001
    alpha = 0.001
    episodes = 100
    order = 5

    results = np.zeros((trials, episodes))
    for trial in range(trials):
        w = np.ones((len(scene.actions), (order+1)**scene.num_state_vars))*init
        results[trial, :] = lm.fncApproxQlearnLam(lam, eps, alpha, episodes, order, w, scene)

    final_results["Mountain Car Q-Learn-Lam"] = results

    # 2.5) Mountain Car w/ FunctionApprox Fourier Actor-Critic
    w_init = 0
    theta_init = 0
    lam = 0.8
    alpha = 0.001
    beta = 0.001
    sigma = 1.0
    episodes = 100
    order = 3

    final_results["Mountain Car Actor-Critic"] = results

    colors = ["#FD8C8C", "#4286F4", "#F4BE41", "#A641F4", "#3FB22A"]
    for (key, result), color in zip(final_results.items(), colors):
        _, _, bars = plt.errorbar(
            x=np.arange(episodes),
            y=result.mean(axis=0),
            yerr=result.std(axis=0), 
            color=color,
            ecolor=color, 
            elinewidth=1,
            label=key)

        [bar.set_alpha(0.4) for bar in bars]

    plt.xlabel("Episode")
    plt.ylabel("Discounted Return")
    plt.ylim(-1000,0)
    plt.yticks(np.arange(-2500,0+1,250))
    plt.title("Mountain Car Method Comparisons")
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()
