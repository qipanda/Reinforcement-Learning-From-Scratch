from typing import Tuple

from environments import Scenario

import numpy as np

def returnEpsGreedyAction(eps: float, q_s: np.ndarray) -> int:
    if np.random.rand() < eps:
        return np.random.randint(len(q_s))
    else:
        return np.random.choice(a=np.flatnonzero(q_s == q_s.max()))

def returnSoftmaxAction(sigma: float, p: np.ndarray, s: int) -> Tuple[int, np.ndarray]:
    softmax_probs = np.exp(sigma*p[s, :])
    softmax_probs = softmax_probs/softmax_probs.sum()
    return np.random.choice(a=p.shape[1], p=softmax_probs), softmax_probs

def returnSoftmaxActionFncApprox(sigma: float, p: np.ndarray) -> Tuple[int, np.ndarray]:
    softmax_probs = np.exp(sigma*p)
    softmax_probs = softmax_probs/softmax_probs.sum()
    return np.random.choice(a=p.size, p=softmax_probs), softmax_probs

def tabularSarsa(eps: float, alpha: float, episodes: int,
                 q: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        a_idx = returnEpsGreedyAction(eps, q[scene.s])
        while scene.s != scene.goal_state:
            s = scene.s
            a = a_idx

            r = scene.takeAction(a_idx)
            a_idx = returnEpsGreedyAction(eps, q[scene.s])
            q[s, a] += alpha*(r + scene.gamma*q[scene.s, a_idx] - q[s, a])

        rewards[ep] = scene.r_cum

    return rewards

def tabularQlearn(eps: float, alpha: float, episodes: int,
                  q: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        while scene.s != scene.goal_state:
            a_idx = returnEpsGreedyAction(eps, q[scene.s])
            s = scene.s
            a = a_idx

            r = scene.takeAction(a_idx)
            q[s, a] += alpha*(r + scene.gamma*q[scene.s, :].max() - q[s, a])

        rewards[ep] = scene.r_cum

    return rewards

def tabularSarsaLam(lam: float, eps: float, alpha: float, episodes: int,
                    q: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        e = np.zeros((scene.num_states, len(scene.actions)))
        a_idx = returnEpsGreedyAction(eps, q[scene.s])
        while scene.s != scene.goal_state:
            s = scene.s
            a = a_idx

            r = scene.takeAction(a_idx)
            a_idx = returnEpsGreedyAction(eps, q[scene.s])
            e *= scene.gamma*lam
            e[s, a] += 1
            delta = r + scene.gamma*q[scene.s, a_idx] - q[s, a]
            q += alpha*delta*e

        rewards[ep] = scene.r_cum

    return rewards

def tabularQlearnLam(lam: float, eps: float, alpha: float, episodes: int,
                     q: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        e = np.zeros((scene.num_states, len(scene.actions)))
        while scene.s != scene.goal_state:
            a_idx = returnEpsGreedyAction(eps, q[scene.s])
            s = scene.s
            a = a_idx

            r = scene.takeAction(a_idx)
            e *= scene.gamma*lam
            e[s, a] += 1
            delta = r + scene.gamma*q[scene.s, :].max() - q[s, a]
            q += alpha*delta*e

        rewards[ep] = scene.r_cum

    return rewards

def tabularActorCritic(lam: float, alpha: float, beta: float, sigma: float, episodes: int,
                        v: np.ndarray, theta: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        e_v = np.zeros(scene.num_states)
        e_theta = np.zeros((scene.num_states, len(scene.actions)))
        while scene.s != scene.goal_state:
            # Act using the actor
            a_idx, softmax_probs = returnSoftmaxAction(sigma, theta, scene.s)
            s = scene.s
            a = a_idx
            r = scene.takeAction(a_idx)

            # Critic update using TD(lam)
            e_v *= scene.gamma*lam
            e_v[s] += 1
            delta = r + scene.gamma*v[scene.s] - v[s]
            v += alpha*delta*e_v

            # Actor update
            e_theta *= scene.gamma*lam
            e_theta[s, :] -= softmax_probs
            e_theta[s, a] += 1
            theta += beta*delta*e_theta

        rewards[ep] = scene.r_cum

    return rewards

def fncApproxSarsa(eps: float, alpha: float, episodes: int, order: int,
                 w: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        phi_s = scene.returnFourierBasis(order)
        q_s = np.array([np.dot(w[i, :], phi_s) for i in range(w.shape[0])])
        a = returnEpsGreedyAction(eps, q_s)
        while np.any(scene.s != scene.goal_state):
            r = scene.takeAction(a)
            phi_s_prime = scene.returnFourierBasis(order)
            q_s_prime = np.array([np.dot(w[i, :], phi_s_prime) for i in range(w.shape[0])])
            a_prime = returnEpsGreedyAction(eps, q_s_prime)
            w[a, :] += alpha*(r + scene.gamma*q_s_prime[a_prime] - q_s[a])*phi_s

            phi_s = phi_s_prime
            q_s = q_s_prime
            a = a_prime

        rewards[ep] = scene.r_cum

    return rewards

def fncApproxQlearn(eps: float, alpha: float, episodes: int, order: int,
                 w: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        phi_s = scene.returnFourierBasis(order)
        q_s = np.array([np.dot(w[i, :], phi_s) for i in range(w.shape[0])])
        a = returnEpsGreedyAction(eps, q_s)
        while np.any(scene.s != scene.goal_state):
            r = scene.takeAction(a)

            phi_s_prime = scene.returnFourierBasis(order)
            q_s_prime = np.array([np.dot(w[i, :], phi_s_prime) for i in range(w.shape[0])])
            a_prime = returnEpsGreedyAction(eps, q_s_prime)

            w[a, :] += alpha*(r + scene.gamma*q_s_prime.max() - q_s[a])*phi_s

            phi_s = phi_s_prime
            q_s = q_s_prime
            a = a_prime

        rewards[ep] = scene.r_cum

    return rewards

def fncApproxSarsaLam(lam: float, eps: float, alpha: float, episodes: int, order: int,
                    w: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        e = np.zeros(w.shape)
        phi_s = scene.returnFourierBasis(order)
        q_s = np.array([np.dot(w[i, :], phi_s) for i in range(w.shape[0])])
        a = returnEpsGreedyAction(eps, q_s)
        while np.any(scene.s != scene.goal_state):
            r = scene.takeAction(a)

            phi_s_prime = scene.returnFourierBasis(order)
            q_s_prime = np.array([np.dot(w[i, :], phi_s_prime) for i in range(w.shape[0])])
            a_prime = returnEpsGreedyAction(eps, q_s_prime)

            e *= scene.gamma*lam
            e[a, :] += phi_s
            delta = r + scene.gamma*q_s_prime[a_prime] - q_s[a]
            w += alpha*delta*e

            phi_s = phi_s_prime
            q_s = q_s_prime
            a = a_prime

        rewards[ep] = scene.r_cum

    return rewards

def fncApproxQlearnLam(lam: float, eps: float, alpha: float, episodes: int, order: int,
                     w: np.ndarray, scene: Scenario) -> np.ndarray:
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        e = np.zeros(w.shape)
        phi_s = scene.returnFourierBasis(order)
        q_s = np.array([np.dot(w[i, :], phi_s) for i in range(w.shape[0])])
        a = returnEpsGreedyAction(eps, q_s)
        while np.any(scene.s != scene.goal_state):
            r = scene.takeAction(a)

            phi_s_prime = scene.returnFourierBasis(order)
            q_s_prime = np.array([np.dot(w[i, :], phi_s_prime) for i in range(w.shape[0])])
            a_prime = returnEpsGreedyAction(eps, q_s_prime)

            e *= scene.gamma*lam
            e[a, :] += phi_s
            delta = r + scene.gamma*q_s_prime.max() - q_s[a]
            w += alpha*delta*e

            phi_s = phi_s_prime
            q_s = q_s_prime
            a = a_prime

        rewards[ep] = scene.r_cum

    return rewards

def fncApproxActorCritic(lam: float, alpha: float, beta: float, sigma: float, episodes: int,
                        order: int, w: np.ndarray, theta: np.ndarray, scene: Scenario) -> np.ndarray:
    num_actions = len(scene.actions)
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        scene.resetState()
        e_v = np.zeros(w.shape)
        e_theta = np.zeros(theta.shape)

        while np.any(scene.s != scene.goal_state):
            phi_s = scene.returnFourierBasis(order)
            p_s = np.array([np.dot(theta[a, :], phi_s) for a in range(num_actions)])
            a, softmax_probs = returnSoftmaxActionFncApprox(sigma, p_s)
            
            # Act using the actor
            r = scene.takeAction(a)

            # Critic update using TD(lam)
            e_v *= scene.gamma*lam
            e_v += phi_s
            delta = r + scene.gamma*np.dot(w, scene.returnFourierBasis(order)) - np.dot(w, phi_s)
            w += alpha*delta*e_v

            # Actor update
            e_theta *= scene.gamma*lam
            e_theta -= softmax_probs.reshape(num_actions,-1)*np.tile(phi_s, num_actions).reshape(num_actions, -1)
            e_theta[a, :] += phi_s
            theta += beta*delta*e_theta

        rewards[ep] = scene.r_cum

    return rewards
