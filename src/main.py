from models.node import Node


def init_grid():
    # TODO: ad some params
    # TODO: set larger than replica amount
    server = Node("Server Node", capacity=99999999, parent=None)

    # TODO: what to return? some Grid object? and Grid.get_server ...



def main():
    print("Main function")

    # XXX: have the main simulation code in a special class?

    init_grid()


    # run simulation

    # print results


if __name__ == "__main__":
    main()
