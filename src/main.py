from models.simulation import Strategy
from models.simulation import Simulation


sim_settings = dict(
    node_capacity=50000,
    strategy=Strategy.EFS,
    replica_count=1000,
)


def main():
    print("Main function")

    # XXX: have the main simulation code in a special class?
    sim = Simulation(**sim_settings)

    sim.init_grid()


    # run simulation

    # print results


if __name__ == "__main__":
    main()
