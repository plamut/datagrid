from collections import OrderedDict
from copy import copy
from models.simulation import Strategy
from models.simulation import Simulation


settings_EFS_1 = dict(
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
    network_bw_mbps=10,  # network bandwidth (megabits)
    pspeed_kmps=6000,  # signal propagation speed
    total_reqs=100000,
    fsti=10000,  # frequency specific time interval
    rnd_seed=1,
)
settings_LFU_1 = copy(settings_EFS_1)
settings_LFU_1['strategy'] = Strategy.LFU
settings_LRU_1 = copy(settings_EFS_1)
settings_LRU_1['strategy'] = Strategy.LRU
settings_MFS_1 = copy(settings_EFS_1)
settings_MFS_1['strategy'] = Strategy.MFS

settings_EFS_2 = copy(settings_EFS_1)
settings_EFS_2['mwg_prob'] = 0.3
settings_LFU_2 = copy(settings_LFU_1)
settings_LFU_2['mwg_prob'] = 0.3
settings_LRU_2 = copy(settings_LRU_1)
settings_LRU_2['mwg_prob'] = 0.3
settings_MFS_2 = copy(settings_MFS_1)
settings_MFS_2['mwg_prob'] = 0.3

settings_EFS_3 = copy(settings_EFS_1)
settings_EFS_3['mwg_prob'] = 0.5
settings_LFU_3 = copy(settings_LFU_1)
settings_LFU_3['mwg_prob'] = 0.5
settings_LRU_3 = copy(settings_LRU_1)
settings_LRU_3['mwg_prob'] = 0.5
settings_MFS_3 = copy(settings_MFS_1)
settings_MFS_3['mwg_prob'] = 0.5

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

    settings = OrderedDict([
        ('SCENARIO 1 [P(mwg) = 0.1]',
            OrderedDict([
                ('LRU', settings_LRU_1),
                ('LFU', settings_LFU_1),
                ('EFS', settings_EFS_1),
                ('MFS', settings_MFS_1)
            ])),
        ('SCENARIO 2 [P(mwg) = 0.3]',
            OrderedDict([
                ('LRU', settings_LRU_2),
                ('LFU', settings_LFU_2),
                ('EFS', settings_EFS_2),
                ('MFS', settings_MFS_2)
            ])),
        ('SCENARIO 3 [P(mwg) = 0.5]',
            OrderedDict([
                ('LRU', settings_LRU_3),
                ('LFU', settings_LFU_3),
                ('EFS', settings_EFS_3),
                ('MFS', settings_MFS_3)
            ])),
    ])

    all_stats = OrderedDict()

    for scenario_name, settings_list in settings.items():
        for strategy_name, sim_settings in settings_list.items():
            sim = Simulation(**sim_settings)
            # XXX: move settings to here? as a parameter to initialize?
            sim.initialize()
            results = sim.run()
            all_stats.setdefault(
                scenario_name, OrderedDict())[strategy_name] = results

    print_results(all_stats)


def print_results(sim_stats):
    """TODO: docstring"""
    bold = '\033[1m'
    reset = '\033[0m'
    yellow_b = '\033[1;33m'

    for scenario_name, sim_run_results in sim_stats.items():
        # header
        print()
        print(yellow_b, 15 * " ", scenario_name, 15 * " ", reset, sep="")
        print("", 10 * "-", 20 * "-", 23 * "-", "", sep="+")
        print(
            "| ", bold, "Strategy", reset, " | ",
            bold, "Total resp. t. [s] ", reset, "| ",
            bold, "Total BW used [Mbit]", reset, "  |", sep="")
        print("", 10 * "-", 20 * "-", 23 * "-", "", sep="+")

        # body
        for strategy_name, results in sim_run_results.items():
            print(
                '|   {}{}{}    |'.format(bold, strategy_name, reset),
                "\t{:9.2f}\t|".format(results['total_resp_time']),
                "\t{:9.2f}\t|".format(results['total_bw']),
                sep='')

        #footer
        print("", 10 * "-", 20 * "-", 23 * "-", "", sep="+")
        print()


if __name__ == "__main__":
    main()
