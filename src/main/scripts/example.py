from networkx import draw_networkx
from pyEpanet.epanet import inp_to_graph, simulate_head_data, prepare_leakages_scenarios
from matplotlib import pyplot as plt

__author__ = 'Alexey Karnachev'

if __name__ == '__main__':
    # Paths:
    inp_path = '../data/Net3_fixed.inp'
    scenarios_folder = '../data/scenarios/'

    # Parse .inp file into networkx graph:
    net = inp_to_graph(inp_path)

    # Visualize the network:
    draw_networkx(net, with_labels=False, pos=net.graph['COORDINATES'], node_size=45)
    plt.show()

    # Run hydraulic simulation and obtain the head data:
    head = simulate_head_data(inp_path)

    # Visualize head data:
    head.plot()
    plt.show()

    # Prepare leakage scenarios:
    prepare_leakages_scenarios(inp_path, leaks_demands=[5, 10, 20], output_folder=scenarios_folder)

    # The scenario folder is now filled with batch of .inp files
