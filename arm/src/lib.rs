// Crates.
use cpython::{Python, PyResult, py_module_initializer, py_fn};
use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

fn arm(py: Python, N: usize, D: usize, E: Array1<f64>, T: f64, R: f64, K: f64,
       S: usize, P: f64, shock: (usize, f64), init: &str, seed: u64)
       -> PyResult<PyTuple> {
    // const N: usize = 100;
    // const D: usize = 1;
    // let E: Array<f64, _> = array![0.1];
    // const T: f64 = 0.25;
    // const R: f64 = 0.25;
    // const K: f64 = f64::INFINITY;
    // const S: usize = 1_000_000;
    // const P: f64 = 0.0;
    // // need to do shock, initialization options
    // const SEED: u32 = 3121127542;

    // Initialize the random number generator.
    let rng = StdRng::seed_from_u64(seed);
    let dist_norm = Normal::new(0.5, 0.2).unwrap();
    let dist_agents = Uniform::from(0..N);
    let dist_uni = Uniform::from(0.0..1.0);

    // Initialize the agent population and their initial ideological positions.
    let mut config: Array2<f64> = Array::zeros((N, D));
    if D == 1 {
        for i in 0..N {
            loop {
                config[[i, 0]] = dist_norm.sample(&mut rng);
                if 0.0 <= config[[i, 0]] && config[[i, 0]] <= 1.0 {
                    break;
                }
            }
        }
    } else {
        panic!("Haven't implemented D > 1 yet.")
    }
    // TODO: Empirical initialization.
    let init_config = config.to_owned();
    println!("{:?}", config);

    // Create an array of historical interactions.
    struct Interaction {
        active: usize,
        passive: usize,
        position: Array1<f64>,
    }
    let mut history = Vec::<Interaction>::new();

    // Simulate the desired number of pairwise interactions.
    for step in 0..S {
        // TODO: external shock.

        // Choose the active agent u.a.r.
        let i = dist_agents.sample(&mut rng);

        // TODO: perform self-interest intervention.

        // Interaction Rule: interact with probability (1/2)^delta, where delta
        // is the decay based on the agents' distance, scaled by the exposures
        // for each dimension.
        let mut j = dist_agents.sample(&mut rng);
        loop {
            if i != j {
                break;
            } else {
                j = dist_agents.sample(&mut rng);
            }
        }
        let mut delta: f64 = 0.0;
        for k in 0..D {
            delta += (config[[i, k]] - config[[j, k]]).powi(2) / E[k].powi(2);
        }
        let delta = delta.sqrt();
        if dist_uni.sample(&mut rng) < (0.5_f64).powf(delta) {
            // The Attraction-Repulsion rule of opinion change.
            let dist = l2_norm(&config.row(i) - &config.row(j));
            if dist <= T {
                // Attraction: agent i moves toward agent j.
                let new_pos = &config.row(i) + R * (&config.row(j) - &config.row(i));
                config.row_mut(i).assign(&new_pos);
            } else {
                // Repulsion: agent i moves away from agent j.
                let new_pos = &config.row(i) - R * (&config.row(j) - &config.row(i));
                config.row_mut(i).assign(&new_pos);
            }
            // Clip to the limits of ideological space.
            for k in 0..D {
                if config[[i, k]] < 0.0 {
                    config.row_mut(i)[k] = 0.0;
                } else if config[[i, k]] > 1.0 {
                    config.row_mut(i)[k] = 1.0;
                }
            }
            // Record the movement.
            history.push(Interaction {
                active: i,
                passive: j,
                position: config.row(i).to_owned(),
            });
        } else {
            // No interaction.
            history.push(Interaction {
                active: i,
                passive: i,
                position: config.row(i).to_owned(),
            });
        }
    }

    println!("{:?}", config);
}

fn l2_norm(x: Array1<f64>) -> f64 {
    x.dot(&x).sqrt()
}

py_module_initializer!(arm, initarm, PyInit_arm, |py, m| {
    m.add(py, "arm", py_fn!(py, arm()));
    Ok(());
});
