# AttractionRepulsionModel

The Attraction-Repulsion Model (ARM) is a simple agent-based model of opinion change developed by Robert Axelrod, Joshua J. Daymude, and Stephanie Forrest.
Its formal description is given in [this paper](https://arxiv.org/abs/2103.06492).
Informally, it has two rules:

1. Actors tend to interact with those who have similar views.

2. Pairwise interactions between similar actors *reduces* their difference while interactions between dissimilar actors *increases* their difference.

This Python implementation simulates the ARM, allowing variations in the following parameters:

| Parameter | Description |
| --- | --- |
| `N` | Number of agents |
| `D` | Number of ideological dimensions |
| `E` | Exposure, the degree to which actors interact with differing points of view |
| `T` | Tolerance, the distance within which interactions are attractive and beyond which interactions are repulsive |
| `R` | Responsiveness, the fractional distance an actor's ideological position moves as a result of an interaction |
| `P` | Probability of self-interest |
| `shock` | External shock timing and strength |


## Getting Started

1. You'll need a command line (Unix-based, Windows Command Prompt, or macOS Terminal) and any Python installation version 3.5 or newer. You will also need the [numpy](https://numpy.org/install/), [matplotlib](https://matplotlib.org/stable/users/installing.html), and [tqdm](https://github.com/tqdm/tqdm#installation) packages.

2. Clone this repository or download the latest [release](https://github.com/jdaymude/AttractionRepulsionModel/releases).

3. Create `data/` and `figs/` in the code directory.

4. To reproduce the data and figures from our paper, run the experiments script:
```
python arm_exp.py -E <id_of_experiment> -R <random_seed>
```
A list of experiments and their IDs can be found in `arm_exp.py`. All results from the paper were obtained with seed `3121127542`.

5. While less convenient, you can alternatively run a single simulation of the ARM with your own parameters of interest:
```
python
>>> from arm import arm
>>> # set up parameters as listed above
>>> # returns the initial configuration, final configuration, and move history
>>> init, final, history = arm(N, D, E, T, R, P, shock)
```

6. Note that the plotting and analysis functions are members of the `Experiment` class in `arm_exp.py`.
If you want to do anything heavier than simply getting the data for a single run, you should add your own experiments to `arm_exp.py` and run them as in Step 4.


## Contributing

If you'd like to leave feedback, feel free to open a [new issue](https://github.com/jdaymude/AttractionRepulsionModel/issues/new/).
If you'd like to contribute, please submit your code via a [pull request](https://github.com/jdaymude/AttractionRepulsionModel/pulls).
