import msprime
import numpy as np
from tqdm import tqdm
from time_discretization import get_interpolated_tmrca_landscape, discretize

np.random.seed(133742)

# msprime parameters
SEQUENCE_LENGTH = 10e6
NUM_SIMULATIONS = 1000
SAMPLES = 2
POPULATION_SIZE = 2e4
RECOMBINATION_RATE = 1e-8
MUTATION_RATE = 1e-8
PLOIDY = 1

# dataset
WINDOW_SIZE = 2000

# population time
TIME_DISC = 256
population_time_log = np.linspace(3, 14, TIME_DISC)

# encode/decode
encode = discretize
decode = lambda i: population_time_log[i]

parameters = {
            'samples': SAMPLES,
            'population_size': POPULATION_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
            'recombination_rate': RECOMBINATION_RATE,
            'ploidy': PLOIDY,
            'random_seed': None}

SEEDS = np.random.randint(1, 1e9, NUM_SIMULATIONS)
tmrcas = []
for seed in tqdm(SEEDS):
    parameters['random_seed'] = seed
    ts = msprime.sim_ancestry(**parameters)
    ts = msprime.mutate(ts, rate=MUTATION_RATE, random_seed=parameters['random_seed'])

    tmrca = np.log(get_interpolated_tmrca_landscape(ts, window_size=WINDOW_SIZE)).tolist()
    tmrcas += tmrca

tokens = encode(tmrcas, population_time_log)
np.save("tokens.npy", tokens)
