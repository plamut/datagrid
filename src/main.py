from models.simulation import Strategy
from models.simulation import Simulation


sim_settings = dict(
    node_count=20,
    node_capacity=50000,
    strategy=Strategy.EFS,
    min_dist_km=1,  # min distance between two adjacent nodes
    max_dist_km=1000,  # max distance between two adjacent nodes
    replica_count=1000,
    replica_min_size=100,  # megabits
    replica_max_size=1000,  # megabits
    rnd_seed=1,
    total_reqs=100000,
    fsti=1000,  # frequency specific time interval
)

debug_sim = dict(
    node_count=6,
    node_capacity=50000,
    strategy=Strategy.EFS,
    min_dist_km=1,  # min distance between two adjacent nodes
    max_dist_km=1000,  # max distance between two adjacent nodes
    replica_count=1000,
    replica_min_size=100,  # megabits
    replica_max_size=1000,  # megabits
    rnd_seed=1,
    total_reqs=5,
    fsti=1000,  # frequency specific time interval
)


def main():
    print("Main function")

    sim = Simulation(**debug_sim)

    # XXX: move settings to here? as a parameter to initialize?
    sim.initialize()
    sim.run()

    # TODO: get results and print them


if __name__ == "__main__":
    main()
