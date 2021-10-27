import numpy as np
from typing import Tuple, List
from collections import defaultdict

from frozen_lake import FrozenLakeEnv


def generate_episode(
    env: FrozenLakeEnv, policy: np.ndarray, render: bool = False
) -> Tuple[List[int], List[int], List[float]]:
    """Generates an episode following the given policy.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        policy (np.ndarray): A 2D array with shape (env.observation_space.n, env.action_space.n) such that each row is
            a probability distribution.
        render (bool, optional): Whether to render the environment at each step of the episode. Defaults to False.

    Returns:
        List[int]: A list of the states visited.
        List[int]: A list of the actions taken.
        List[float]: A list of the rewards received.
    """
    env.reset()
    done = False
    states = []
    actions = []
    rewards = []

    while not done:
        if render:
            env.render()
        
        states.append(env.s)
        a = np.random.choice(policy.shape[1], p=policy[env.s])
        actions.append(a)

        next_state, reward, done, extra = env.step(a)
        rewards.append(reward)

    return states, actions, rewards


def uniform_policy(observation_space_size: int, action_space_size: int) -> np.ndarray:
    """Generates a uniform policy over all actions of each state.

    Args:
        observation_space_size (int): Number of states.
        action_space_size (int): Number of actions per state.

    Returns:
        np.ndarray: A uniform policy.
    """
    policy = np.ones((observation_space_size, action_space_size))
    policy /= policy.sum(axis=1, keepdims=True)
    return policy


def first_visit_epsilon_soft_mc_control(
    env: FrozenLakeEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    epsilon: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs on-policy first-visit Monte-Carlo control for epsilon-soft policies to estimate the optimal policy. The 
    policy is initialized as a uniform policy and the Q value of each state-action pair is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.
        epsilon (float, optional): Controls the degree of exploration. Should be a value in [0, 1]. Defaults to 0.05.

    Returns:
        np.ndarray: Estimated Q-values of each state-action pair.
        np.ndarray: Estimated optimal policy.
        np.ndarray: Cumulated returns for each episode.
    """
    policy = uniform_policy(env.observation_space.n, env.action_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Optimistic initialization of all state-action returns with one return of 0
    state_action_returns = {
        (s, a): [0]  # Index by (state, action) tuple
        for s in range(env.observation_space.n)
        for a in range(env.action_space.n)
    }
    episode_returns = []

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(env, policy)
        episode_returns.append(sum(rewards))

        # Iterate over a set to avoid duplicate state-action pairs
        for s, a in set((_s, _a) for (_s, _a) in zip(states, actions)):
            # Calling next on the generator yields only the first value
            first_occurence_idx = next(i for i, (_s, _a) in enumerate(zip(states, actions)) if _s == s and _a == a)
            G = sum(r * gamma ** i for i, r in enumerate(rewards[first_occurence_idx:]))
            state_action_returns[s, a].append(G)
            Q[s, a] = np.mean(state_action_returns[s, a])

            # Update policy of state to epsilon-greedy
            p = np.ones(env.action_space.n) * epsilon / env.action_space.n
            p[np.argmax(Q[s])] += 1 - epsilon
            policy[s] = p

    return Q, policy, np.asarray(episode_returns)


def ordinary_importance_sampling_mc_prediction(
    env: FrozenLakeEnv,
    target: np.ndarray,
    n_episodes: int = 2000,
    gamma: float = 0.99,
) -> np.ndarray:
    """Runs (ordinary) importance sampling Monte Carlo prediction to estimate the value function V(s) of the target
    policy. The behaviour policy is the a uniform policy over all actions. The Q value of each state-action pair is 
    initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        target (np.ndarray): The target policy whose values we want to estimate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.

    Returns:
        np.ndarray: An array of shape (n_episodes, env.observation_space.n) containing the value estimates for each
            state after every episode.
    """
    behaviour = uniform_policy(env.observation_space.n, env.action_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    N = np.zeros((env.observation_space.n, env.action_space.n))
    Vs = []

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(env, behaviour)
        G = 0
        W = 1

        for s, a, r in zip(reversed(states), reversed(actions), reversed(rewards)):
            G = gamma * G + r
            N[s, a] += 1
            Q[s, a] += W * (G - Q[s, a]) / N[s, a]
            W *= target[s, a] / behaviour[s, a]

        Vs.append(np.sum(target * Q, axis=1))

    return np.stack(Vs)


def weighted_importance_sampling_mc_prediction(
    env: FrozenLakeEnv,
    target: np.ndarray,
    n_episodes: int = 2000,
    gamma: float = 0.99,
) -> np.ndarray:
    """Runs weighted importance sampling Monte Carlo prediction to estimate the value function V(s) of the target 
    policy. The behaviour policy is the a uniform policy over all actions. The Q value of each state-action pair is 
    initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        target (np.ndarray): The target policy whose values we want to estimate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.

    Returns:
        np.ndarray: An array of shape (n_episodes, env.observation_space.n) containing the value estimates for each
            state after every episode.
    """
    behaviour = uniform_policy(env.observation_space.n, env.action_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    C = np.zeros((env.observation_space.n, env.action_space.n))
    Vs = []

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(env, behaviour)

        G = 0
        W = 1
        for s, a, r in zip(reversed(states), reversed(actions), reversed(rewards)):
            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += W / C[s, a] * (G - Q[s, a])
            W *= target[s, a] / behaviour[s, a]

        Vs.append(np.sum(target * Q, axis=1))

    return np.stack(Vs)


def every_visit_mc_prediction(
    env: FrozenLakeEnv,
    policy: np.ndarray,
    n_episodes: int = 10000,
    gamma: float = 0.99,
) -> np.ndarray:
    """Runs every-visit Monte Carlo prediction to estimate the value function V(s) of the given policy. The value of 
    each state is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        policy (np.ndarray): The policy whose values we want to estimate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 10000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.

    Returns:
        np.ndarray: An array of shape (n_episodes, env.observation_space.n) containing the value estimates for each
            state after every episode.
    """
    state_returns = defaultdict(list)
    V = np.zeros(env.observation_space.n)
    Vs = []

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(env, policy)
        
        G = 0
        for s, a, r in zip(reversed(states), reversed(actions), reversed(rewards)):
            G = gamma * G + r
            state_returns[s].append(G)
            V[s] = np.mean(state_returns[s])
        
        Vs.append(V.copy())
    
    return np.stack(Vs)


def td_0_prediction(
    env: FrozenLakeEnv,
    policy: np.ndarray,
    n_episodes: int = 10000,
    gamma: float = 0.99,
    alpha: float = 0.01,
):
    """Runs the TD(0) prediction algorithm to estimate the value function V(s) of the given policy. The value of 
    each state is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        policy (np.ndarray): The policy whose values we want to estimate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 10000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.
        alpha (float, optional): Step-size of the value updates. Defaults to 0.01.

    Returns:
        np.ndarray: An array of shape (n_episodes, env.observation_space.n) containing the value estimates for each
            state after every episode.
    """
    V = np.zeros(env.observation_space.n)
    Vs = []

    for _ in range(n_episodes):
        env.reset()
        done = False

        while not done:
            state = env.s
            a = np.random.choice(policy.shape[1], p=policy[state])
            next_state, reward, done, extra = env.step(a)
            V[state] += alpha * (reward + gamma*V[next_state] - V[state])
        
        Vs.append(V.copy())
    
    return np.stack(Vs)


def n_step_td_prediction(
    env: FrozenLakeEnv,
    policy: np.ndarray,
    n: int,
    n_episodes: int = 10000,
    gamma: float = 0.99,
    alpha: float = 0.01,
) -> np.ndarray:
    """Runs the n-step TD prediction algorithm to estimate the value function V(s) of the given policy. The value of 
    each state is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        policy (np.ndarray): The policy whose values we want to estimate.
        n (int): Number of bootstrapping steps.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.
        alpha (float, optional): Step-size of the value updates. Defaults to 0.01.

    Returns:
        np.ndarray: An array of shape (n_episodes, env.observation_space.n) containing the value estimates for each
            state after every episode.
    """
    V = np.zeros(env.observation_space.n)
    Vs = []

    for _ in range(n_episodes):
        env.reset()
        states = [env.s]
        rewards = [0]

        t = 0
        T = float('inf')

        while True:
            t += 1

            if t < T:
                a = np.random.choice(policy.shape[1], p=policy[env.s])
                next_state, reward, done, extra = env.step(a)
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t

            tau = t - n
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(T, tau + n) + 1):
                    G += pow(gamma, i - tau - 1) * rewards[i]

                if tau + n <= T:
                    G += pow(gamma, n) * V[states[(tau + n)]]

                V[states[tau]] += alpha * (G - V[states[tau]])
            
            if tau == T - 1:
                break
    
        Vs.append(V.copy())

    return np.stack(Vs)


def modified_n_step_td_prediction(
    env: FrozenLakeEnv,
    policy: np.ndarray,
    n: int,
    n_episodes: int = 10000,
    gamma: float = 0.99,
) -> np.ndarray:
    """Runs a modified version of the n-step TD prediction algorithm to estimate the value function V(s) of the given
    policy, such that the estimation is equivalent to every-visit Monte Carlo prediction. Instead of a constant step
    size $\\alpha$, the step size is changed to $\\alpha_t = 1/N(S_t)$ where $N(S_t)$ is the number of times the state
    $S_t$ has been visited. The value of each state is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        policy (np.ndarray): The policy whose values we want to estimate.
        n (int): Number of bootstrapping steps.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.

    Returns:
        np.ndarray: An array of shape (n_episodes, env.observation_space.n) containing the value estimates for each
            state after every episode.
    """
    N = np.zeros(env.observation_space.n)
    V = np.zeros(env.observation_space.n)
    Vs = []

    for _ in range(n_episodes):
        env.reset()
        states = [env.s]
        rewards = [0]

        t = 0
        T = float('inf')

        while True:
            t += 1

            if t < T:
                a = np.random.choice(policy.shape[1], p=policy[env.s])
                next_state, reward, done, extra = env.step(a)
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t

            tau = t - n
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(T, tau + n) + 1):
                    G += pow(gamma, i - tau - 1) * rewards[i]

                if tau + n <= T:
                    G += pow(gamma, n) * V[states[(tau + n)]]

                # TODO: change alpha s.t. it decays with the number of visits
                N[states[tau]] += 1
                V[states[tau]] += (G - V[states[tau]]) / N[states[tau]]
            
            if tau == T - 1:
                break
    
        Vs.append(V.copy())

    return np.stack(Vs)


def sarsa(
    env: FrozenLakeEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the SARSA control algorithm to estimate the Q values. The policy is initialized as a uniform policy and the
    Q value of each state-action pair is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.
        alpha (float, optional): Step size parameter. Defaults to 0.1.
        epsilon (float, optional): Controls the degree of exploration. Should be a value in [0, 1]. Defaults to 0.01.

    Returns:
        np.ndarray: Estimated Q-values of each state-action pair.
        np.ndarray: Estimated optimal epsilon-greedy policy.
        np.ndarray: Cumulated returns for each episode.
    """
    # Policy is initialized as uniform, and then becomes epsilon-greedy as training advances
    policy = uniform_policy(env.observation_space.n, env.action_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    episode_returns = []

    for _ in range(n_episodes):
        episode_returns.append(0)
        state = env.reset()
        action = np.random.choice(policy.shape[1], p=policy[state])
        done = False
        while not done:
            next_state, reward, done, extra = env.step(action)
            next_action = np.random.choice(policy.shape[1], p=policy[next_state])
            Q[state, action] += alpha * (reward + gamma*Q[next_state, next_action] - Q[state, action])

            # Update policy of state to epsilon-greedy
            p = np.ones(env.action_space.n) * epsilon / env.action_space.n
            p[np.argmax(Q[state])] += 1 - epsilon
            policy[state] = p

            action, state = next_action, next_state
            episode_returns[-1] += reward
    
    return Q, policy, np.asarray(episode_returns)


def expected_sarsa(
    env: FrozenLakeEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    alpha: float = 0.2,
    epsilon: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the expected SARSA control algorithm to estimate the Q values. The policy is initialized as a uniform policy
    and the Q value of each state-action pair is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.
        alpha (float, optional): Step size parameter. Defaults to 0.2.
        epsilon (float, optional): Controls the degree of exploration. Should be a value in [0, 1]. Defaults to 0.01.

    Returns:
        np.ndarray: Estimated Q-values of each state-action pair.
        np.ndarray: Estimated optimal epsilon-greedy policy.
        np.ndarray: Cumulated returns for each episode.
    """
    # Policy is initialized as uniform, and then becomes epsilon-greedy as training advances
    policy = uniform_policy(env.observation_space.n, env.action_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    episode_returns = []

    for _ in range(n_episodes):
        episode_returns.append(0)
        state = env.reset()
        action = np.random.choice(policy.shape[1], p=policy[state])
        done = False
        while not done:
            next_state, reward, done, extra = env.step(action)
            next_action = np.random.choice(policy.shape[1], p=policy[next_state])
            Q[state, action] += alpha * (reward + gamma*np.sum(policy[next_state]*Q[next_state]) - Q[state, action])

            # Update policy of state to epsilon-greedy
            p = np.ones(env.action_space.n) * epsilon / env.action_space.n
            p[np.argmax(Q[state])] += 1 - epsilon
            policy[state] = p

            action, state = next_action, next_state
            episode_returns[-1] += reward
    
    return Q, policy, np.asarray(episode_returns)


def q_learning(
    env: FrozenLakeEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the Q-learning control algorithm to estimate the Q values. The policy is initialized as a uniform policy and
    the Q value of each state-action pair is initialized to 0.

    Args:
        env (FrozenLakeEnv): The frozen lake environment to navigate.
        n_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        gamma (float, optional): Discount factor. Should be a value in [0, 1]. Defaults to 0.99.
        alpha (float, optional): Step size parameter. Defaults to 0.1.
        epsilon (float, optional): Controls the degree of exploration. Should be a value in [0, 1]. Defaults to 0.02.

    Returns:
        np.ndarray: Estimated Q-values of each state-action pair.
        np.ndarray: Estimated optimal epsilon-greedy policy.
        np.ndarray: Cumulated returns for each episode.
    """
    # Policy is initialized as uniform, and then becomes epsilon-greedy as training advances
    policy = uniform_policy(env.observation_space.n, env.action_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    episode_returns = []

    for _ in range(n_episodes):
        episode_returns.append(0)
        state = env.reset()
        action = np.random.choice(policy.shape[1], p=policy[state])
        done = False
        while not done:
            next_state, reward, done, extra = env.step(action)
            next_action = np.random.choice(policy.shape[1], p=policy[next_state])
            Q[state, action] += alpha * (reward + gamma*Q[next_state].max() - Q[state, action])

            # Update policy of state to epsilon-greedy
            p = np.ones(env.action_space.n) * epsilon / env.action_space.n
            p[np.argmax(Q[state])] += 1 - epsilon
            policy[state] = p

            action, state = next_action, next_state
            episode_returns[-1] += reward
    
    return Q, policy, np.asarray(episode_returns)
