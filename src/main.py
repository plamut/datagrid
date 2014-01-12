
from copy import copy
from models.simulation import Strategy
from models.simulation import Simulation


settings_EFS = dict(
    node_count=20,
    node_capacity=50000,
    strategy=Strategy.EFS,
    min_dist_km=1,  # min distance between two adjacent nodes
    max_dist_km=1000,  # max distance between two adjacent nodes
    replica_count=1000,
    replica_group_count=10,
    mwg_prob=0.1,  # probability of requesting a replica from MWG
    replica_min_size=100,  # megabits
    replica_max_size=1000,  # megabits
    rnd_seed=1,
    total_reqs=100000,
    fsti=10000,  # frequency specific time interval
)

settings_LFU = copy(settings_EFS)
settings_LFU['strategy'] = Strategy.LFU

settings_LRU = copy(settings_EFS)
settings_LRU['strategy'] = Strategy.LRU

# debugging configuration
settings_dbg = dict(
    node_count=6,
    node_capacity=50000,
    strategy=Strategy.EFS,
    min_dist_km=1,  # min distance between two adjacent nodes
    max_dist_km=1000,  # max distance between two adjacent nodes
    replica_count=1000,
    replica_group_count=10,
    replica_min_size=100,  # megabits
    replica_max_size=1000,  # megabits
    rnd_seed=1,
    total_reqs=5,
    fsti=1000,  # frequency specific time interval
)


def main():

    sim = Simulation(**settings_EFS)

    # XXX: move settings to here? as a parameter to initialize?
    sim.initialize()
    sim.run()

    # TODO: get results and print them


if __name__ == "__main__":
    main()
