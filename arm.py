# Project:     AttractionRepulsionModel
# Filename:    arm.py
# Authors:     Joshua J. Daymude (jdaymude@asu.edu).

"""
arm: An agent-based model of ideological polarization utilizing both attractive
     and repulsive interactions.
"""

import math
import numpy as np
from tqdm import trange


def arm(N=100, D=1, E=[0.1], T=0.25, R=0.25, S=500000, P=0, shock=(None, None),\
        seed=None, silent=False):
    """
    Execute a simulation of the Attraction-Repulsion Model.

    Inputs:
    N (int): number of agents
    D (int): number of ideological dimensions
    E ([float]): list of exposures
    T (float): tolerance
    R (float): responsiveness
    S (int): number of steps to simulate
    P (float): self-interest probability
    shock ((float, float)): external shock step and strength
    seed (int): random seed
    silent (bool): True if progress should be shown on command line

    Returns (init_config, config, history):
    init_config: N x D array of initial agent ideological positions
    config: N x D array of agent ideological positions after S steps
    history: S x (D + 2) array detailing interaction history
    """

    # Initialize the random number generation.
    rng = np.random.default_rng(seed)

    # Initialize the agent population and their initial ideological positions.
    if D == 1:
        config = np.zeros(N)
        for i in np.arange(N):
            while True:
                config[i] = rng.normal(0.5, 0.2)
                if 0 <= config[i] and config[i] <= 1:
                    break
        config = config.reshape(-1, 1)
    else:  # Higher dimensions.
        means, covs = 0.5 + np.zeros(D), 0.04 * np.eye(D)
        config = np.zeros((N, D))
        for i in np.arange(N):
            while True:
                config[i] = rng.multivariate_normal(means, covs)
                clip = np.maximum(np.zeros(D), np.minimum(np.ones(D), config[i]))
                if np.allclose(config[i], clip):
                    break
    init_config = np.copy(config)

    # Create an S x (D + 2) array to store the interaction history. Each step i
    # records the active agent [i][0], the passive agent [i][1], and the active
    # agent's new position [i][2:].
    history = np.zeros((S, D + 2))

    # Simulate the desired number of pairwise interactions.
    for step in trange(S, desc='Simulating interactions', disable=silent):
        # Perform the external shock intervention, if specified.
        shock_step, shock_strength = shock
        if shock_step is not None:
            assert D == 1, 'ERROR: External shock requires 1D'
            if step >= shock_step and step < shock_step + N:
                i = step - shock_step
                config[i] = np.minimum(np.ones(D), config[i] + shock_strength)
                history[step] = np.concatenate(([i], [i], config[i]))
                continue

        # Choose the active agent u.a.r.
        i = rng.integers(N)

        # Perform self-interest intervention, if specified.
        if P > 0 and rng.random() < P:
            config[i] = config[i] + R * (init_config[i] - config[i])
            history[step] = np.concatenate(([i], [i], config[i]))
            continue

        # Interaction Rule: interact with probability (1/2)^d, where d is the
        # decay based on the agents' distance, scaled by the exposures for each
        # dimension.
        j = rng.choice(np.delete(np.arange(N), i))
        d = math.sqrt(sum([(config[i][k] - config[j][k])**2 / \
                            E[k]**2 for k in range(D)]))
        if rng.random() <= math.pow(0.5, d):
            # The Attraction-Repulsion rule of opinion change.
            if np.linalg.norm(config[i] - config[j]) <= T:
                # Attraction: agent i moves toward agent j.
                config[i] = config[i] + R * (config[j] - config[i])
            else:
                # Repulsion: agent i moves away from agent j.
                config[i] = config[i] - R * (config[j] - config[i])
            # Clip to the limits of ideological space.
            config[i] = np.maximum(np.zeros(D), np.minimum(D, config[i]))
            history[step] = np.concatenate(([i], [j], config[i]))
        else:  # No interaction.
            history[step] = np.concatenate(([i], [i], config[i]))

    return init_config, config, history
