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
    fste=1000,  # frequency specific time interval
)


def main():
    print("Main function")

    # XXX: have the main simulation code in a special class?
    sim = Simulation(**sim_settings)

    sim.initialize()


    # run simulation

    # print results


if __name__ == "__main__":
    main()
