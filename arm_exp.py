# Project:     AttractionRepulsionModel
# Filename:    arm_exp.py
# Authors:     Joshua J. Daymude (jdaymude@asu.edu).

"""
arm_exp: A framework for defining and running experiments for the
         Attraction-Repulsion Model.
"""

import argparse
from arm import arm
from itertools import product
import pickle
import math
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Experiment(object):
    """
    A flexible, unifying framework for experiments.
    """

    def __init__(self, id, params={}, iters=1, savehist=True, seed=None):
        """
        Inputs:
        id (str): identifier for the experiment, e.g., 'A' or 'baseline'
        params (dict): the full parameter set for the simulation runs
            {
              'N' : [int: > 1],
              'D' : [int: > 0],
              'E' : [[float: > 0]],
              'T' : [float: > 0 and < sqrt(D)],
              'R' : [float: > 0 and <= 1],
              'K' : [float: > 1],
              'S' : [int: > 0],
              'P' : [float: >= 0 and <= 1],
              'shock' : [(int: >= 0, float: >= 0 and <= 1)],
              'init' : ['norm', 'emp']
            }
        iters (int): the number of iterated runs for each parameter setting
        savehist (bool): True if a run's history should be saved
        seed (int): random seed
        """
        # Unpack singular parameters.
        self.id, self.iters, self.savehist, self.seed, = id, iters, savehist, seed

        # Unpack ARM parameters.
        defaults = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : [0.25], \
                    'R' : [0.25], 'K' : [math.inf], 'S' : [500000], 'P' : [0], \
                    'shock' : [(None, None)], 'init' : ['norm']}
        plist = [params[p] if p in params else defaults[p] for p in defaults]
        self.params = list(product(*plist))

        # Set up data and results filenames.
        self.fname = 'exp{}_{}'.format(self.id, self.seed)

        # Instantiate a list to hold runs data. This data will have shape
        # A x B x [N x D, N x D, S x (D + 2)] where A is the number of runs
        # (i.e., unique parameter combinations); B is the number of iterations
        # per run; and N, D, S are as they are in the polarization framework.
        self.runs_data = [[] for p in self.params]


    def run(self):
        tqdm.write('Running Experiment ' + self.id + '...')

        # Set up random seeds for iterated runs.
        rng = np.random.default_rng(self.seed)
        run_seeds = rng.integers(0, 2**32, size=self.iters)

        # For each parameter combination, do iterated runs of polarization.
        silent = len(self.params) > 1 or self.iters > 1
        for i, param in enumerate(tqdm(self.params, desc='Simulating runs')):
            N, D, E, T, R, K, S, P, shock, init = param
            for seed in tqdm(run_seeds, desc='Iterating run', \
                             leave=bool(i == len(self.params) - 1)):
                run_data = arm(N, D, E, T, R, K, S, P, shock, init, seed, silent)
                if not self.savehist:
                    self.runs_data[i].append((run_data[0], run_data[1], []))
                else:
                    self.runs_data[i].append(run_data)


    def save(self):
        """
        Saves this experiment, including all parameters and run data, to a file
        named according to the experiment's ID and seed.
        """
        tqdm.write('Saving Experiment ' + self.id + '...')
        with open('data/' + self.fname + '.pkl', 'wb') as f:
            pickle.dump(self, f)


    def variance(self, config):
        """
        Takes as input an N x D configuration and returns its variance.
        """
        return sum(np.var(config, axis=0))


    def variances(self, run, iter):
        """
        Takes as input a run and iteration index and returns a 1 x S array of
        variances of the agents' ideological positions at each step.
        """
        # Check that a configuration history exists.
        assert self.savehist, 'ERROR: No history to calculate variance per step.'

        # Get the initial configuration and move history; initialize variances.
        config, _, moves = self.runs_data[run][iter]
        config = np.copy(config)  # Avoid editing actual data.
        vars = np.zeros(len(moves) + 1)

        # Replay the agent movements one step at a time and calculate variance.
        vars[0] = self.variance(config)
        for step, move in enumerate(moves):
            config[int(move[0])] = move[2:]
            vars[step+1] = self.variance(config)

        return vars


    def plot_evo(self, runs, iters):
        """
        Takes indices of either (i) one run and multiple iterations or (ii) one
        iteration of multiple runs and plots the given metrics against time.
        """
        tqdm.write('Plotting variance over time...')

        # Sanity checks and setup.
        assert self.savehist, 'ERROR: No history to calculate metrics per step.'
        assert len(runs) == 1 or len(iters) == 1, 'ERROR: One run or one iter'
        runits = [i for i in product(runs, iters)]

        # Set up colors.
        cmap = np.vectorize(lambda x : cm.plasma(x))
        colors = np.array(cmap(np.linspace(0, 0.9, len(runits)))).T

        # Plot variance over time for each run/iteration.
        fig, ax = plt.subplots()
        for i, runit in enumerate(tqdm(runits, desc='Calculating variance')):
            y = self.variances(runit[0], runit[1])
            ax.plot(np.arange(len(y)), y, color=colors[i])
        ax.set(xlabel='# Steps', ylabel='Variance')
        ax.grid()
        plt.tight_layout()
        fig.savefig('figs/' + self.fname + '.png', dpi=300)
        plt.close()


    def plot_sweep(self, p1, p2, plabels, runs, cmax=None):
        """
        Plots the average variance for each run's iterations as a 2D color mesh,
        where the mesh is organized according to the given parameter ranges.
        """
        tqdm.write('Plotting average variance...')

        # Calculate average variance per run.
        aves = np.zeros(len(runs))
        for i, run in enumerate(tqdm(runs, desc='Averaging iterations')):
            aves[i] = np.average([self.variance(iter[1]) \
                                  for iter in self.runs_data[run]])

        # Plot average variances.
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(p1, p2, aves.reshape(len(p1), len(p2)).T, \
                            cmap='plasma', vmin=0, vmax=cmax, shading='nearest')
        fig.colorbar(pcm, ax=ax, label='Variance')
        ax.set(xlabel=plabels[0], ylabel=plabels[1])
        plt.tight_layout()
        fig.savefig('figs/' + self.fname +'.png', dpi=300)
        plt.close()


    def animate_1D(self, run, iter, frame=None, anno='', colormode=None):
        """
        Animate a 1D histogram of agent ideological positions.
        """
        tqdm.write('Animating histogram of ideological positions...')

        # Check that a configuration history exists.
        assert self.savehist, 'ERROR: No history to show cliques per step.'

        config, _, moves = self.runs_data[run][iter]
        config = np.copy(config)  # Avoid editing actual data.
        S, N, D = len(moves) + 1, np.shape(config)[0], self.params[run][1]
        assert D == 1, 'ERROR: Can only animate 1D'

        # Set up colors.
        cmap = np.vectorize(lambda x : cm.plasma(x))
        if colormode == 0:  # Color corresponding to the run number.
            c = np.array(cmap(np.linspace(0, 0.9, len(self.params)))).T[run]
        elif colormode == 1:  # Color corresponding to the iteration number.
            c = np.array(cmap(np.linspace(0, 0.9, self.iters))).T[iter]
        else:  # Use black no matter what.
            c = 'k'

        # Set up plot and histogram.
        fig, ax = plt.subplots(dpi=300)
        bins = 50
        hist, edges = np.histogram(config, bins=bins, range=[0, 1])
        bar = ax.bar(edges[:-1], hist, width=1/bins, align='edge', color=c)

        def init():
            ax.set_title('step 0 of {}'.format(S-1), loc='right', \
                         fontsize='small')
            ax.set(xlabel='D1', ylabel='# Agents', xlim=[0, 1], ylim=[0, N])
            ax.grid()
            plt.tight_layout()
            return [b for b in bar]

        # Set frame step.
        if frame == None:
            # Target 50fps with duration that scales linearly with steps.
            secs = (11 / 49600) * S + (565 / 62)
            frame_step = int(S / (50 * secs))
        else:
            frame_step = frame

        def update(i):
            # Replay the configuration's move history for the elapsed time.
            if i > 0:
                for step in np.arange(i - frame_step + 1, i + 1):
                    config[int(moves[step-1][0])] = moves[step-1][2:]

            # Update the figure.
            ax.set_title('step {} of {}'.format(i, S-1), loc='right', \
                         fontsize='small')
            hist, _ = np.histogram(config, bins=bins, range=[0, 1])
            [b.set_height(hist[bi]) for bi, b in enumerate(bar)]

            return [b for b in bar]

        # Animate.
        frames = np.arange(0, S, frame_step)
        ani = FuncAnimation(fig, update, frames, init, interval=20, blit=True)
        fname = 'figs/' + self.fname + '_ani'
        fname += '_' + anno if (anno != '') else ''
        ani.save(fname + '.mp4')
        plt.close()


    def animate_2D(self, run, iter, frame=None, anno=''):
        """
        Animate a 2D histogram of agent ideological positions.
        """
        tqdm.write('Animating histogram of ideological positions...')

        # Check that a configuration history exists.
        assert self.savehist, 'ERROR: No history to show cliques per step.'

        config, _, moves = self.runs_data[run][iter]
        config = np.copy(config)  # Avoid editing actual data.
        S, N, D = len(moves) + 1, np.shape(config)[0], self.params[run][1]
        assert D == 2, 'ERROR: Can only animate 2D'

        fig, ax = plt.subplots(dpi=300)
        num_bins = [50, 50]
        hist, xedges, yedges = \
            np.histogram2d(config.T[0], config.T[1], bins=num_bins, \
                           range=[[0, 1], [0, 1]])
        X, Y = np.meshgrid(xedges, yedges)
        pcm = ax.pcolormesh(X, Y, hist.T, cmap='plasma', norm=LogNorm(1, N))
        fig.colorbar(pcm, ax=ax, label='# Agents')

        def init():
            ax.set_title('step 0 of {}'.format(S-1), loc='right', \
                         fontsize='small')
            ax.set(xlabel='D1', ylabel='D2')
            plt.tight_layout()
            return pcm,

        # Set frame step.
        if frame == None:
            # Target 50fps with duration that scales linearly with steps.
            secs = (11 / 49600) * S + (565 / 62)
            frame_step = int(S / (50 * secs))
        else:
            frame_step = frame

        def update(i):
            # Replay the configuration's move history for the elapsed time.
            if i > 0:
                for step in np.arange(i - frame_step + 1, i + 1):
                    config[int(moves[step-1][0])] = moves[step-1][2:]

            # Update the figure.
            ax.set_title('step {} of {}'.format(i, S-1), loc='right', \
                         fontsize='small')
            hist = np.histogram2d(config.T[0], config.T[1], bins=num_bins, \
                                  range=[[0, 1], [0, 1]])[0]
            pcm.set_array(hist.T)

            return pcm,

        # Animate.
        frames = np.arange(0, S, frame_step)
        ani = FuncAnimation(fig, update, frames, init, interval=20, blit=True)
        fname = 'figs/' + self.fname + '_ani'
        fname += '_' + anno if (anno != '') else ''
        ani.save(fname + '.mp4')
        plt.close()


def expA_evo(seed=None):
    """
    With default parameters in 1D and a subset of tolerance-responsiveness
    space, investigate the system's evolution w.r.t. variance.

    Data from this experiment produces Figs. 1 and 2.
    """
    T = np.arange(0.05, 1.01, 0.1)
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : T, 'R' : [0.25], \
              'K' : [math.inf], 'S' : [2500000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('A_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expA_sweep(seed=None):
    """
    With default parameters in 1D, sweep tolerance-responsiveness space and plot
    average final variance.

    Data from this experiment produces Fig. 3.
    """
    T, R = np.arange(0.05, 1.01, 0.05), np.arange(0.05, 1.01, 0.05)
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : T, 'R' : R, \
              'K' : [math.inf], 'S' : [1000000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('A_sweep', params, iters=20, savehist=False, seed=seed)
    exp.run()
    exp.save()
    exp.plot_sweep(T, R, ('T', 'R'), runs=np.arange(len(exp.params)), cmax=0.25)


def expB_evo(seed=None):
    """
    With default parameters in 1D and a subset of tolerance-exposure space,
    investigate the system's evolution w.r.t. variance.

    Data from this experiment produces Fig. 5.
    """
    E = [[e] for e in np.arange(0.05, 0.51, 0.05)]
    params = {'N' : [100], 'D' : [1], 'E' : E, 'T' : [0.3], 'R' : [0.25], \
              'K' : [math.inf], 'S' : [2500000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('B_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expB_sweep(seed=None):
    """
    With default parameters in 1D, sweep tolerance-exposure space and plot
    average final variance.

    Data from this experiment produces Fig. 4.
    """
    T, E = np.arange(0.05, 1.01, 0.05), [[e] for e in np.arange(0.05, 0.51, 0.05)]
    params = {'N' : [100], 'D' : [1], 'E' : E, 'T' : T, 'R' : [0.25], \
              'K' : [math.inf], 'S' : [2000000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('B_sweep', params, iters=20, savehist=False, seed=seed)
    exp.run()
    exp.save()
    exp.plot_sweep(T, E, ('T', 'E'), runs=np.arange(len(exp.params)), cmax=0.25)


def expC_evo(seed=None):
    """
    With default parameters in 2D and a subset of tolerance-responsiveness
    space, investigate the system's evolution w.r.t. variance.

    Data from this experiment was not used in the paper.
    """
    T = np.arange(0.05, 2**0.5, 0.1)
    params = {'N' : [100], 'D' : [2], 'E' : [[0.1, 0.1]], 'T' : T, \
              'R' : [0.25], 'K' : [math.inf], 'S' : [2500000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('C_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expC_sweep(seed=None):
    """
    With default parameters in 2D, sweep tolerance-responsiveness space and plot
    average final variance.

    Data from this experiment produces Fig. S2.
    """
    T, R = np.arange(0.05, 2**0.5, 0.05), np.arange(0.05, 1.01, 0.05)
    params = {'N' : [100], 'D' : [2], 'E' : [[0.1, 0.1]], 'T' : T, 'R' : R, \
              'K' : [math.inf], 'S' : [1000000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('C_sweep', params, iters=20, savehist=False, seed=seed)
    exp.run()
    exp.save()
    exp.plot_sweep(T, R, ('T', 'R'), runs=np.arange(len(exp.params)), cmax=0.5)


def expD_evo(seed=None):
    """
    With default parameters in 2D and a subset of exposures, investigate the
    system's evolution w.r.t. variance.

    Data from this experiment produces Fig. 6.
    """
    E = [[0.1, e] for e in np.arange(0.05, 0.51, 0.05)]
    params = {'N' : [100], 'D' : [2], 'E' : E, 'T' : [0.25], 'R' : [0.25], \
              'K' : [math.inf], 'S' : [2500000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('D_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expD_sweep(seed=None):
    """
    With default parameters in 2D, sweep exposures and plot average final
    variance.

    Data from this experiment produces Fig. S3.
    """
    E = np.arange(0.05, 0.51, 0.05)
    params = {'N' : [100], 'D' : [2], 'E' : list(product(E, E)), 'T' : [0.25], \
              'R' : [0.25], 'K' : [math.inf], 'S' : [2000000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('D_sweep', params, iters=20, savehist=False, seed=seed)
    exp.run()
    exp.save()
    exp.plot_sweep(E, E, ('E1', 'E2'), runs=np.arange(len(exp.params)), cmax=0.5)


def expE_evo(seed=None):
    """
    With default parameters in 1D and a subset of self-interest space,
    investigate the system's evolution w.r.t. variance.

    Data from this experiment produces Fig. 7.
    """
    P = np.arange(0, 0.11, 0.01)
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : [0.25], \
              'R' : [0.25], 'K' : [math.inf], 'S' : [2500000], 'P' : P, \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('E_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expE_sweep(seed=None):
    """
    With default parameters in 1D, sweep self-interest space and plot average
    final variance.

    Data from this experiment produces Fig. S4.
    """
    T, P = np.arange(0.05, 1.01, 0.1), np.arange(0, 1.001, 0.05)
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : [0.25], \
              'R' : [0.25], 'K' : [math.inf], 'S' : [2000000], 'P' : P, \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('E_sweep', params, iters=20, savehist=False, seed=seed)
    exp.run()
    exp.save()
    exp.plot_sweep(T, P, ('T', 'P'), runs=np.arange(len(exp.params)), cmax=0.25)


def expF_evo(seed=None):
    """
    With default parameters in 1D and a subset of external shocks, investigate
    the system's evolution w.r.t. variance.

    Data from this experiment produces Fig. 8.
    """
    shocks = [(500000, delta) for delta in np.arange(0, 0.81, 0.05)]
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : [0.25], \
              'R' : [0.25], 'K' : [math.inf], 'S' : [2500000], 'P' : [0], \
              'shock' : shocks, 'init' : ['norm']}
    exp = Experiment('F_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expF_sweep(seed=None):
    """
    With default parameters in 1D, sweep external shocks and plot final average
    variance.

    Data from this experiment produces Fig. 9.
    """
    steps, deltas = np.arange(100000, 900001, 100000), np.arange(0, 0.81, 0.05)
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : [0.25], \
              'R' : [0.25], 'K' : [math.inf], 'S' : [2000000], 'P' : [0], \
              'shock' : list(product(steps, deltas)), 'init' : ['norm']}
    exp = Experiment('F_sweep', params, iters=20, savehist=False, seed=seed)
    exp.run()
    exp.save()
    exp.plot_sweep(steps, deltas, ('Shock Step', 'Shock Strength'), \
                   runs=np.arange(len(exp.params)), cmax=0.25)


def expR1_storep(seed=None):
    """
    With default parameters in 1D and varying steepness of stochastic repulsion,
    investigate the system's evolution w.r.t. variance.

    Data from this experiment produces Fig. S1.
    """
    K = np.append([np.power(2, i) for i in np.arange(1, 7)], [math.inf])
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : [0.25], \
              'R' : [0.25], 'K' : K, 'S' : [2000000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['norm']}
    exp = Experiment('R1_storep', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


def expR1_emp_evo(seed=None):
    """
    With default parameters in 1D, a subset of tolerance-responsiveness
    space, and empirical initialization, investigate the system's evolution
    w.r.t. variance.

    Data from this experiment produces Fig. ??.
    """
    T = np.arange(0.05, 1.01, 0.1)
    params = {'N' : [100], 'D' : [1], 'E' : [[0.1]], 'T' : T, 'R' : [0.25], \
              'K' : [math.inf], 'S' : [2500000], 'P' : [0], \
              'shock' : [(None, None)], 'init' : ['emp']}
    exp = Experiment('R1_emp_evo', params, seed=seed)
    exp.run()
    exp.save()
    exp.plot_evo(runs=np.arange(len(exp.params)), iters=[0])


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-E', '--exps', type=str, nargs='+', required=True, \
                        help='IDs of experiments to run')
    parser.add_argument('-R', '--rand_seed', type=int, default=None, \
                        help='Seed for random number generation')
    args = parser.parse_args()

    # Run selected experiments.
    exps = {'A_evo' : expA_evo, 'A_sweep' : expA_sweep, 'B_evo' : expB_evo, \
            'B_sweep' : expB_sweep, 'C_evo' : expC_evo, 'C_sweep' : expC_sweep,\
            'D_evo' : expD_evo, 'D_sweep' : expD_sweep, 'E_evo' : expE_evo, \
            'E_sweep' : expE_sweep, 'F_evo' : expF_evo, 'F_sweep' : expF_sweep,\
            'R1_storep' : expR1_storep, 'R1_emp_evo' : expR1_emp_evo}
    for id in args.exps:
        exps[id](args.rand_seed)
