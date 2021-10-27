from typing import Optional
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Directions(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def plot_policy(
    policy: np.ndarray,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cbar_label: str = "Probability",
):
    """Plots the policy as a heat map, with a color bar. This function can also be used to plot state or state-action
    values.

    Args:
        policy (np.ndarray): A 2D array of the policy 
        fig (plt.Figure, optional): A plt.Figure object in which the policy is plotted. If none is provided, a new one 
            is generated. Defaults to None.
        ax (plt.Axes, optional): A plt.Axes object  in which the policy is plotted. If none is provided, a new one is 
            generated. Defaults to None.
        cbar_label (str, optional): The label string for the color bar. Defaults to "Probability".
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(policy)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=f"{100 / policy.shape[1]}%", pad=0.1)
    fig.colorbar(im, cax=cax, orientation="vertical", label=cbar_label)
    ax.set_xticks(range(policy.shape[1]))
    ax.set_xticklabels([Directions(i).name for i in range(policy.shape[1])], rotation="vertical")
    ax.set_yticks(range(policy.shape[0]))
    ax.set_xlabel("Actions")
    ax.set_ylabel("States")


def plot_many(experiments: np.ndarray, ax: Optional[plt.Axes] = None, label: Optional[str] = None):
    """Plots the experiments' mean with a shaded fill of +/- the standard deviation.

    Args:
        experiments (np.ndarray): A 2D array with shape (n_runs, n_episodes) containing the experimental results.
        ax (plt.Axes, optional): A plt.Axes object in which the results are plotted. If none is provided, a new one is 
            generated. Defaults to None.
        label (str, optional): The label string of the experiments. Defaults to None.
    """
    if ax is None:
        _, ax = plt.subplots()

    mean_exp = np.mean(experiments, axis=0)
    std_exp = np.std(experiments, axis=0)
    ax.plot(mean_exp, label=label)
    ax.fill_between(range(len(experiments[0])), mean_exp + std_exp, mean_exp - std_exp, alpha=0.1)


def plot_all_state_values(Vs: np.ndarray, ax: Optional[plt.Axes] = None):
    """Plots the value over time of all state values in `Vs`.

    Args:
        Vs (np.ndarray): A 2D array with shape (n_episodes, env.observation_space.n) containing the state values over 
            time.
        ax (plt.Axes, optional): A plt.Axes object in which the values are plotted. If none is provided, a new one is 
            generated. Defaults to None.
    """
    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1.0, Vs.shape[1]))

    for s, (V, c) in enumerate(zip(Vs.T, colors)):
        ax.plot(V, label=s, color=c, alpha=0.8)
        ax.legend(ncol=4, loc="best")
