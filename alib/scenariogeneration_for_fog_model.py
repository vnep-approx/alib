# MIT License
#
# Copyright (c) 2019 Balazs Nemeth
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from . import scenariogeneration as sg, datamodel
import networkx as nx


class ABBUseCaseRequestGenerator(sg.AbstractRequestGenerator):
    """
    Generates a industrial fog application described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems,
    Parameters are made more accurate from Suter's MSc thesis at ETH Zuric.
    """

    EXPECTED_PARAMETERS = [
        'sensor_actuator_loop_count' # If not even, one lower number is used
    ]

    def __init__(self, logger=None):
        super(ABBUseCaseRequestGenerator).__init__(logger=logger)
        # All parameters of the request generator are inicialized here and the same names are expected in 'raw_parameters'
        self.sensor_actuator_loop_count = None
        self.universal_node_type = 'universal'

    def _read_raw_parameters(self, raw_parameters):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            self.sensor_actuator_loop_count = int(raw_parameters['sensor_actuator_loop_count'])
            if self.sensor_actuator_loop_count % 2 == 1:
                self.sensor_actuator_loop_count -= 1
        except KeyError as e:
            raise sg.ExperimentSpecificationError("Parameter not found in request specification: {keyerror}".format(keyerror=e))

    def _add_single_preprocessing_block(self, req, index, substrate_node):
        """
        Adds a single A-B-C-D-E block of preprocessor network to req with index. Returns the nodes where the rest of the application
        should be connected

        :param substrate_node:
        :param req:
        :return:
        """
        index_str = str(index)
        for preproc_node, demand in zip(["A", "B", "C", "D", "E"],
                                          [27.55, 26.65, 9.6, 3.6, 15.25]):
            req.add_node(preproc_node + index_str, demand=demand, ntype=self.universal_node_type)
        for sensor_actuator_node, demand in zip(["S", "T"], [2.0, 0.25]):
            req.add_node(sensor_actuator_node + index_str, demand=demand, ntype=self.universal_node_type, allowed_nodes=[substrate_node])
        req.add_edge("S"+index_str, "A"+index_str, demand=3072)
        req.add_edge("A"+index_str, "B"+index_str, demand=384)
        req.add_edge("B"+index_str, "C"+index_str, demand=448)
        req.add_edge("C"+index_str, "D"+index_str, demand=256)
        req.add_edge("C"+index_str, "E"+index_str, demand=128)

        return "E" + index_str, "D" + index_str, "T" + index_str

    def generate_request(self, name, raw_parameters, substrate):
        """
        Realizes the generator function to fit to the framework.

        :param name:
        :param raw_parameters:
        :param substrate:
        :return:
        """
        self._read_raw_parameters(raw_parameters)
        req = datamodel.Request("ABB_fog_app")
        for base_node, demand in zip(["F", "G", "H"], [20.875, 20,875, 14.5]):
            req.add_node(base_node, demand=demand, ntype=self.universal_node_type)
        req.add_edge("F", "H", demand=192)
        req.add_edge("G", "H", demand=192)
        # NOTE: with 'sample' we assume all actuator - sensor pairs are on different infrastucture nodes
        nodes_for_actuators_sensors = sg.random.sample(substrate.nodes(), self.sensor_actuator_loop_count)
        for index in xrange(1, self.sensor_actuator_loop_count + 1):
            nodeE, nodeD, nodeT = self._add_single_preprocessing_block(req, index, nodes_for_actuators_sensors[index - 1])

            # select where to connect the preprocessing block
            if index <= self.sensor_actuator_loop_count / 2:
                node_for_aggregation = "F"
            else:
                node_for_aggregation = "G"

            # connect upward edges
            req.add_edge(nodeE, node_for_aggregation, demand=96)
            req.add_edge(nodeD, node_for_aggregation, demand=160)
            req.add_edge("H", nodeT, demand=68)

            #connect backward edges
            req.add_edge("H", nodeE, demand=68)
            req.add_edge("H", nodeD, demand=68)

        return req


class CactusGraphGenerator(object):

    def __init__(self, n, cycle_tree_ratio, cycle_count_ratio, tree_count_ratio, random=None):
        """
        Creates a datamodel.Graph, without setting any parameters of resource or a request.

        :param n:
        :param cycle_tree_ratio:
        :param cycle_count_ratio:
        :param tree_count_ratio:
        """
        if random is None:
            self.random = sg.random
        else:
            self.random = random
        self.G = datamodel.Graph("cactus")
        self.n = n
        self.cycle_tree_ratio = cycle_tree_ratio
        self.cycle_count_ratio = cycle_count_ratio
        self.tree_count_ratio = tree_count_ratio
        cycle_N = int(n * cycle_tree_ratio)
        self.cycle_node_count = max(int(cycle_N * cycle_count_ratio), 3)
        self.tree_node_count = max(int((n - cycle_N) * tree_count_ratio), 2)

    def _get_node_seq(self, current_edges, remaining_nodes, length):
        """


        :param current_edges:
        :param remaining_nodes:
        :param length:
        :return:
        """
        actual_length = min(len(remaining_nodes), length)
        if len(current_edges) == 0:
            node_seq = self.random.sample(remaining_nodes, actual_length)
        else:
            # get a random edge's head
            _, starting_node = self.random.choice(current_edges)
            node_seq = [starting_node] + self.random.sample(remaining_nodes, actual_length - 1)
        return actual_length, node_seq

    def _add_tree_to_req(self, current_edges, remaining_nodes):
        """
        Adds a tree to the graph, using built in networkX function based on Pruefer sequence to generate a random tree.
        The tree contains elements from the remaining nodes.

        :param current_edges:
        :param remaining_nodes:
        :return:
        """
        node_count, node_seq = self._get_node_seq(current_edges, remaining_nodes, self.tree_node_count)
        # this uses the Pruefer sequence, as Suter's MSc thesis, in the FAAP simulator
        nx_tree =nx.random_tree(node_count, self.random)
        # the networkx tree is undirected, we arbitrarily direct each edge
        new_edges = []
        for i,j in nx_tree.edges():
            # we handle the tree node name as index
            tail = node_seq[i]
            head = node_seq[j]
            self.G.add_edge(tail, head)
            new_edges.append((tail, head))

        return new_edges

    def _add_cycle_to_req(self, current_edges, remaining_nodes):
        """
        Adds a cycle to the request, containing elements from the remaining nodes.

        :param current_edges:
        :param remaining_nodes:
        :return:
        """
        node_count, node_seq = self._get_node_seq(current_edges, remaining_nodes, self.cycle_node_count)
        new_edges = []
        if node_count > 2:
            for i, j in zip(node_seq[:-1], node_seq[1:]):
                self.G.add_edge(i, j)
                new_edges.append((i,j))
            # the last edge
            self.G.add_edge(node_seq[0], node_seq[-1])
            new_edges.append((node_seq[0], node_seq[-1]))

        return new_edges

    def generate_cactus(self):
        """
        Creates the cactus graph.

        :return:
        """
        for i in range(0, self.n):
            self.G.add_node(i)
        remaining_nodes = set(self.G.nodes)
        current_edges = set()
        while len(remaining_nodes) != 0:
            if self.random.random() < 0.5:
                new_edges = self._add_tree_to_req(list(current_edges), list(remaining_nodes))
            else:
                new_edges = self._add_cycle_to_req(list(current_edges), list(remaining_nodes))
            for i,j in new_edges:
                current_edges.add((i,j))
                if i in remaining_nodes:
                    remaining_nodes.remove(i)
                if j in remaining_nodes:
                    remaining_nodes.remove(j)
            # only one remaining node cannot be handled by either function
            if len(remaining_nodes) == 1:
                if not remaining_nodes.issubset(self.G.nodes):
                    tail = self.random.choice(list(self.G.nodes))
                    head = remaining_nodes.pop()
                    current_edges.add((tail, head))
                    # adding a single unconnected node cannot ruin the cactus property.
                    self.G.add_edge(tail, head)
                else:
                    remaining_nodes.pop()
        return self.G


class ABBUseCaseFogNetworkGenerator(sg.ScenariogenerationTask):
    """
    Generates a industrial fog network described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems
    """

    EXPECTED_PARAMETERS = [
        # NOTE: framework does not allow communication of these parameter between request an substrate
        'sensor_actuator_loop_count' # If not even, one lower number is used
    ]

    def __init__(self, logger=None):
        super(ABBUseCaseFogNetworkGenerator).__init__(logger=logger)
        self.sensor_actuator_loop_count = None

    def _read_raw_parameters(self, raw_parameters):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            self.sensor_actuator_loop_count = int(raw_parameters['sensor_actuator_loop_count'])
            if self.sensor_actuator_loop_count % 2 == 1:
                self.sensor_actuator_loop_count -= 1
        except KeyError as e:
            raise sg.ExperimentSpecificationError("Parameter not found in request specification: {keyerror}".format(keyerror=e))

    def apply(self, scenario_parameters, scenario):
        class_raw_parameters_dict = scenario_parameters[sg.SUBSTRATE_GENERATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise sg.ScenarioGeneratorError("No class name found in config file.")
        raw_parameters = class_raw_parameters_dict[class_name]
        self._read_raw_parameters(raw_parameters)
        substrate = datamodel.Substrate("ABB_fog_net")

        scenario.substrate = substrate