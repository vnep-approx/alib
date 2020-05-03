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

import networkx as nx
import math

from . import scenariogeneration as sg, datamodel, util


class ABBUseCaseRequestGenerator(sg.AbstractRequestGenerator):
    """
    Generates a industrial fog application described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems,
    Parameters are made more accurate from Suter's MSc thesis at ETH Zuric.
    """

    EXPECTED_PARAMETERS = [
        # 'sensor_actuator_loop_count'  # referred to as 'N' in the article. Must be given for the substrate
        'normalize',                  # used by the base class during apply. Should be set to False, because we have absolute values in
                                      # the use case
        'number_of_requests',             # used by the base class during apply.
        'exclude_sensor_locations'      # sets the allowed nodes for non sensors so  they cannot be allocated where the sensors are (False by default)
    ]

    def __init__(self, logger=None):
        super(ABBUseCaseRequestGenerator, self).__init__(logger=logger)
        # All parameters of the request generator are inicialized here and the same names are expected in 'raw_parameters'
        self.sensor_actuator_loop_count = None
        self.universal_node_type = 'universal'

    def _read_raw_parameters(self, raw_parameters, substrate):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            # The framework does not allow such communication between the request and the substrate by the config files
            self.sensor_actuator_loop_count = getattr(substrate, 'sensor_actuator_loop_count')
            if 'exclude_sensor_locations' in raw_parameters:
                self.exclude_sensor_locations = bool(raw_parameters['exclude_sensor_locations'])
            else:
                self.exclude_sensor_locations = False
        except Exception as e:
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
        allowed_nodes = None
        if self.exclude_sensor_locations:
            allowed_nodes = list(self.non_sensor_locations)
        for preproc_node, demand in zip(["A", "B", "C", "D", "E"],
                                          [27.55, 26.65, 9.6, 3.6, 15.25]):
            req.add_node(preproc_node + index_str, demand=demand, ntype=self.universal_node_type, allowed_nodes=allowed_nodes)
        for sensor_actuator_node, demand in zip(["S", "T"], [2.0, 0.25]):
            req.add_node(sensor_actuator_node + index_str, demand=demand, ntype=self.universal_node_type, allowed_nodes=[substrate_node])
        req.add_edge("S"+index_str, "A"+index_str, demand=3072)
        req.add_edge("A"+index_str, "B"+index_str, demand=384)
        req.add_edge("B"+index_str, "C"+index_str, demand=448)
        req.add_edge("C"+index_str, "D"+index_str, demand=256)
        req.add_edge("C"+index_str, "E"+index_str, demand=128)
        self.logger.debug("Added preprocessing block # {} to be allocated on substrate {}".format(index, substrate_node))

        return "E" + index_str, "D" + index_str, "T" + index_str

    def generate_request(self, name, raw_parameters, substrate):
        """
        Realizes the generator function to fit to the framework.

        :param name:
        :param raw_parameters:
        :param substrate:
        :return:
        """
        self._read_raw_parameters(raw_parameters, substrate)
        req = datamodel.Request("ABB_" + name)
        nodes_for_actuators_sensors = [i for i in xrange(substrate.get_number_of_nodes() - self.sensor_actuator_loop_count,
                                                         substrate.get_number_of_nodes())]
        self.non_sensor_locations = [u for u in substrate.nodes if u not in nodes_for_actuators_sensors]
        allowed_nodes = None
        if self.exclude_sensor_locations:
            allowed_nodes = list(self.non_sensor_locations)
        for base_node, demand in zip(["F", "G", "H"], [20.875, 20,875, 14.5]):
            req.add_node(base_node, demand=demand, ntype=self.universal_node_type, allowed_nodes=allowed_nodes)
            self.logger.debug("Added F, G, H nodes")
        req.add_edge("F", "H", demand=192)
        req.add_edge("G", "H", demand=192)
        # NOTE: might be implemented nicer (but more difficultly with placement resctriction generation
        # we assume all actuator - sensor pairs are on different infrastucture nodes
        # The location bound nodes are the highest sensor_actuator_loop_count number of nodes.

        # NOTE: checking if the substrate node is chosen from the right set, in our scenario, only these can have a capacity of 2.26.
        for node_id in nodes_for_actuators_sensors:
            if substrate.node[node_id]['capacity'][self.universal_node_type] != 2.26:
                raise Exception("Wrong substrate node would be chosen for location bound node, based on capacity convention!")
        self.logger.debug("Using substrate nodes {} as sensor - actuar locations".format(nodes_for_actuators_sensors))
        for index in xrange(1, self.sensor_actuator_loop_count + 1):
            nodeE, nodeD, nodeT = self._add_single_preprocessing_block(req, index, nodes_for_actuators_sensors[index - 1])

            # select where to connect the preprocessing block
            if index <= self.sensor_actuator_loop_count / 2:
                node_for_aggregation = "F"
            else:
                node_for_aggregation = "G"

            self.logger.debug("Connecting preprocessing block to node {}".format(node_for_aggregation))
            # connect upward edges
            req.add_edge(nodeE, node_for_aggregation, demand=96)
            req.add_edge(nodeD, node_for_aggregation, demand=160)
            req.add_edge("H", nodeT, demand=68)

            #connect backward edges
            req.add_edge("H", nodeE, demand=68)
            req.add_edge("H", nodeD, demand=68)

        return req


class CactusGraphGenerator(object):

    def __init__(self, n, cycle_tree_ratio, cycle_count_ratio, tree_count_ratio, random=None, logger=None):
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
        cycle_N = n * cycle_tree_ratio
        self.cycle_node_count = max(int(cycle_N * cycle_count_ratio), 3)
        self.tree_node_count = max(int((n - cycle_N) * tree_count_ratio), 2)
        if logger is None:
            self.logger = util.get_logger("CactusGraphGenerator", make_file=False)
        else:
            self.logger = logger

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
        self.logger.debug("Added tree edges: {}".format(new_edges))
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
        self.logger.debug("Added cycles edges: {}".format(new_edges))
        return new_edges

    def generate_cactus(self):
        """
        Creates the cactus graph.

        :return:
        """
        self.logger.info("Generating cactus graph of size: {}".format(self.n))
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
                node_list = list(self.G.nodes)
                head = remaining_nodes.pop()
                node_list.remove(head)
                if len(node_list) == 0:
                    break
                tail = self.random.choice(node_list)
                current_edges.add((tail, head))
                # adding a single unconnected node cannot ruin the cactus property.
                self.G.add_edge(tail, head)
        return self.G


class ABBUseCaseFogNetworkGenerator(sg.ScenariogenerationTask):
    """
    Generates a industrial fog network described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems
    """

    EXPECTED_PARAMETERS = [
        # NOTE: framework does not allow communication of these parameter between request an substrate
        'sensor_actuator_loop_count'        # Referred to as 'N'. Should be the same as in the request
        'cycle_tree_ratio',                 # cactus generation param       = 0.6
        'cycle_count_ratio',                # cactus generation param       = 0.2
        'tree_count_ratio',                 # cactus generation param       = 0.2
        'q_capacity_ratio',                 # fog network spare capacity factor after allocation = 2
        'fog_device_utilization_discount'   # background utilization        = 0.5 (50%)
        'pseudo_random_seed',               # used for all randomization actions
        'have_at_least_one_large',           # add at least one computation node with 'Large' capacity
        'node_cost'                             # optional, 0.0 by default
    ]

    def __init__(self, logger=None):
        super(ABBUseCaseFogNetworkGenerator, self).__init__(logger)
        self.sensor_actuator_loop_count = None
        self.random = sg.random
        self.universal_node_type = 'universal'
        self.large_capacity_node_added = False

    def _read_raw_parameters(self, raw_parameters):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            self.logger.debug("Reading configuration from raw parameters: {}".format(raw_parameters))
            self.sensor_actuator_loop_count = int(raw_parameters['sensor_actuator_loop_count'])
            self.cycle_tree_ratio = float(raw_parameters['cycle_tree_ratio'])
            self.cycle_count_ratio = float(raw_parameters['cycle_count_ratio'])
            self.tree_count_ratio = float(raw_parameters['tree_count_ratio'])
            self.q_capacity_ratio = float(raw_parameters['q_capacity_ratio'])
            self.fog_device_utilization_discount = float(raw_parameters['fog_device_utilization_discount'])
            self.have_at_least_one_large = bool(raw_parameters['have_at_least_one_large'])
            if 'pseudo_random_seed' in raw_parameters:
                self.random.seed(int(raw_parameters['pseudo_random_seed']))
            if 'node_cost' in raw_parameters:
                self.node_cost = float(raw_parameters['node_cost'])
            else:
                self.node_cost = 0.0
        except KeyError as e:
            raise sg.ExperimentSpecificationError("Parameter not found in request specification: {keyerror}".format(keyerror=e))

    def _calculate_cactus_order(self):
        """
        Creates the fog network size according to the constants of the Fog application. See details in Suther's thesis.

        :return:
        """
        return int(math.ceil((84.9 * self.sensor_actuator_loop_count + 56.25) /
                         (1725.0 * self.fog_device_utilization_discount) *
                          self.q_capacity_ratio))

    def _get_fog_node_capacity(self):
        """
        Reproduces the resource distribution of the fog network, as described in the ABB Use Case.

        :return:
        """
        # TODO: with the randomization, and even with the 'have_at_least_one_large' set to True there is a small ratio of feasibility
        # return 100000
        P = self.random.random()
        if P < 0.6:
            # small IoT devices
            return self.random.uniform(50, 200)
        elif P < 0.8:
            # medium devices
            return self.random.uniform(500, 2000)
        else:
            # large devices
            self.large_capacity_node_added = True
            return self.random.uniform(4000, 10000)

    def apply(self, scenario_parameters, scenario):
        """
        Framework interface to implement. Extends the scenario with the substrate network.

        :param scenario_parameters:
        :param scenario:
        :return:
        """
        class_raw_parameters_dict = scenario_parameters[sg.SUBSTRATE_GENERATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise sg.ScenarioGeneratorError("No class name found in config file.")
        raw_parameters = class_raw_parameters_dict[class_name]
        self._read_raw_parameters(raw_parameters)
        substrate = datamodel.Substrate("ABB_fog_net")

        # instantiate the class and calculate the cactus right away
        cactus_graph = CactusGraphGenerator(n=self._calculate_cactus_order(),
                                            cycle_tree_ratio=self.cycle_tree_ratio,
                                            cycle_count_ratio=self.cycle_count_ratio,
                                            tree_count_ratio=self.tree_count_ratio,
                                            random=self.random, logger=self.logger).\
                        generate_cactus()
        self.logger.info("Using generated cactus to construct fog network")
        # copy the cactus structure
        for n in cactus_graph.nodes:
            fog_node_capacity = self._get_fog_node_capacity()
            # 0 cost for node resources, as described in the fog allocation paper
            substrate.add_node(n, types=[self.universal_node_type], capacity={self.universal_node_type: fog_node_capacity},
                               cost={self.universal_node_type: self.node_cost})
        if self.have_at_least_one_large and not self.large_capacity_node_added:
            # 'n' will always have some value, the cactus graph cannot have 0 nodes
            self.logger.info("Setting the last substrate node's {} capacity to be large, because none was added!".format(n))
            substrate.node[n]['capacity'][self.universal_node_type] = self.random.uniform(4000, 10000)
        for i,j in cactus_graph.edges:
            # Substrate links are bidirectional by default!!
            # unit cost gives the minimization for hopcount as in the fog allocation paper
            # link capacity of 10 000 is a new addition to the ABB use case since the Fog allocation paper.
            substrate.add_edge(i, j, capacity=10000, cost=1.0)
        # + self.sensor_actuator_loop_count number of nodes for the location bounds.
        non_location_nodes = list(substrate.nodes)
        first_location_node_id = substrate.get_number_of_nodes()
        for n in range(first_location_node_id,
                       first_location_node_id + self.sensor_actuator_loop_count):
            # capacity only for the sensor and actuator
            substrate.add_node(n, types=[self.universal_node_type], capacity={self.universal_node_type: 2.26},
                               cost={self.universal_node_type: 0.0})
            connecting_node = self.random.choice(non_location_nodes)
            # add undirected connection to location
            substrate.add_edge(connecting_node, n, capacity=10000, cost=1.0)
            self.logger.debug("Connecting sensor - actuator substrate node {} to cactus node {}".format(n, connecting_node))

        # NOTE: this is ugly but we have no other choice to communicate between the request and the substrate the same number of sensor_actuator_loop_count
        setattr(substrate, 'sensor_actuator_loop_count', self.sensor_actuator_loop_count)

        # bind the substrate to the scenario
        scenario.substrate = substrate


class SyntheticCactusSubstrateGenerator(sg.ScenariogenerationTask):
    """
    Generates a synthetic fog network described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems
    """

    EXPECTED_PARAMETERS = [
        'node_count',                   # graph order of the cactus
        'cycle_tree_ratio',             # cactus generation param       = 0.6
        'cycle_count_ratio',            # cactus generation param       = 0.2
        'tree_count_ratio',             # cactus generation param       = 0.2
        'node_capacity_interval',       # interval of uniform distribution of node capacity = [0, 1]
        'link_capacity_interval'        # link capacity interval for uniform distribution = [0, 1]
        'pseudo_random_seed',
        'node_cost'                     # optional, 0.0 by default
    ]

    def __init__(self, logger=None):
        super(SyntheticCactusSubstrateGenerator, self).__init__(logger)
        self.random = sg.random
        self.universal_node_type = 'universal'

    def _read_raw_parameters(self, raw_parameters):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            self.logger.debug("Reading configuration from raw parameters: {}".format(raw_parameters))
            self.node_count = int(raw_parameters['node_count'])
            self.cycle_tree_ratio = float(raw_parameters['cycle_tree_ratio'])
            self.cycle_count_ratio = float(raw_parameters['cycle_count_ratio'])
            self.tree_count_ratio = float(raw_parameters['tree_count_ratio'])
            node_capacity_interval = list(raw_parameters['node_capacity_interval'])
            self.node_min_capacity = float(node_capacity_interval[0])
            self.node_max_capacity = float(node_capacity_interval[1])

            link_capacity_interval = list(raw_parameters['link_capacity_interval'])
            self.link_min_capacity = float(link_capacity_interval[0])
            self.link_max_capacity = float(link_capacity_interval[1])
            if 'node_cost' in raw_parameters:
                self.node_cost = float(raw_parameters['node_cost'])
            else:
                self.node_cost = 0.0
            if 'pseudo_random_seed' in raw_parameters:
                self.random.seed(int(raw_parameters['pseudo_random_seed']))
        except KeyError as e:
            raise sg.ExperimentSpecificationError("Parameter not found in request specification: {keyerror}".format(keyerror=e))

    def apply(self, scenario_parameters, scenario):
        """
        Framework interface to implement. Extends the scenario with the substrate network.

        :param scenario_parameters:
        :param scenario:
        :return:
        """
        class_raw_parameters_dict = scenario_parameters[sg.SUBSTRATE_GENERATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise sg.ScenarioGeneratorError("No class name found in config file.")
        raw_parameters = class_raw_parameters_dict[class_name]
        self._read_raw_parameters(raw_parameters)
        substrate = datamodel.Substrate("fog_net")

        # instantiate the class and calculate the cactus right away
        cactus_graph = CactusGraphGenerator(n=self.node_count,
                                            cycle_tree_ratio=self.cycle_tree_ratio,
                                            cycle_count_ratio=self.cycle_count_ratio,
                                            tree_count_ratio=self.tree_count_ratio,
                                            random=self.random, logger=self.logger).\
                        generate_cactus()
        self.logger.info("Using generated cactus to construct fog network")
        # copy the cactus structure
        for n in cactus_graph.nodes:
            fog_node_capacity = self.random.uniform(self.node_min_capacity, self.node_max_capacity)
            # 0 cost for node resources, as described in the fog allocation paper
            substrate.add_node(n, types=[self.universal_node_type], capacity={self.universal_node_type: fog_node_capacity},
                               cost={self.universal_node_type: self.node_cost})
        for i,j in cactus_graph.edges:
            link_capacity = self.random.uniform(self.link_min_capacity, self.link_max_capacity)
            # links are bidirectional by default!!
            # unit cost gives the minimization for hopcount as in the fog allocation paper
            substrate.add_edge(i, j, capacity=link_capacity, cost=1.0)

        # bind the substrate to the scenario
        scenario.substrate = substrate


class SyntheticSeriesParallelDecomposableRequestGenerator(sg.AbstractRequestGenerator):
    """
    Generates a synthetic fog application described in
    Suter, Eidenbenz, Pignolet, Singla -- Fog Application Allocation for Automation Systems,
    Parameters are made more accurate from Suter's MSc thesis at ETH Zuric.
    """

    EXPECTED_PARAMETERS = [
        'request_substrate_node_count_ratio',               # factor of how much more app nodes than substrate nodes        = 2
        'node_demand_interval',                              # interval of uniform distribution of node demand           = [0, 1/3]
        'link_demand_interval',                              # interval of uniform distribution of link resource demand           = [0, 1/2]
        'parallel_serial_ratio',                             # ratio of parallel and serial decompositions                                    = 0.5
        'range_splitter',                                    # ratio of recursive split of node numbers for subgraphs to be composed         = 0.5
                                                             # OR the ratio of series and parallel decompositions if use_connected_sp_def is True
        'location_bound_mapping_ratio',                      # ratio of location bound app nodes to total number of app nodes                = 0.1
        'normalize',                                        # used by the base class during apply. Should be set to False, because we have absolute values in
                                                            # the use case
        'number_of_requests',                                # used by the base class during apply.
        'pseudo_random_seed',
        'use_connected_sp_def'                               # True by default, use Series parallel graph instead of SPD
    ]

    def __init__(self, logger=None):
        super(SyntheticSeriesParallelDecomposableRequestGenerator, self).__init__(logger=logger)
        # All parameters of the request generator are inicialized here and the same names are expected in 'raw_parameters'
        self.universal_node_type = 'universal'
        self.random = sg.random
        self.current_node_id = 0
        self.range_splitter = None
        self.parallel_serial_ratio = None
        self.substrate_nodes_with_bounded_app_node = set()

    def _read_raw_parameters(self, raw_parameters, substrate):
        """
        Reads all expected parameters

        :param raw_parameters:
        :return:
        """
        try:
            self.logger.debug("Reading configuration from raw parameters: {}".format(raw_parameters))
            if 'use_connected_sp_def' in raw_parameters:
                self.use_connected_sp_def = bool(raw_parameters['use_connected_sp_def'])
            else:
                self.use_connected_sp_def = True
            self.number_of_requests = int(raw_parameters['number_of_requests'])
            self.request_substrate_node_count_ratio = float(raw_parameters['request_substrate_node_count_ratio'])
            self.node_count = int(substrate.get_number_of_nodes() * self.request_substrate_node_count_ratio)
            node_demand_interval = list(raw_parameters['node_demand_interval'])
            self.min_node_demand = float(node_demand_interval[0])
            self.max_node_demand = float(node_demand_interval[1])
            link_demand_interval = list(raw_parameters['link_demand_interval'])
            self.min_link_demand = float(link_demand_interval[0])
            self.max_link_demand = float(link_demand_interval[1])
            self.parallel_serial_ratio = float(raw_parameters['parallel_serial_ratio'])
            self.range_splitter = float(raw_parameters['range_splitter'])
            if self.range_splitter != 0.5 and not self.use_connected_sp_def:
                self.logger.warn("Series parallel decomposable graph generation is not guaranteed to terminate by the used definition "
                                 "with other 'range_splitter' value than 0.5!")
            self.location_bound_mapping_ratio = float(raw_parameters['location_bound_mapping_ratio'])
            if 'pseudo_random_seed' in raw_parameters:
                self.random.seed(int(raw_parameters['pseudo_random_seed']))
        except Exception as e:
            raise sg.ExperimentSpecificationError("Parameter not found in request specification: {keyerror}".format(keyerror=e))

    def series_parallel_decomposable_generator(self, n):
        """
        Generates a series parallel decomposable graph by the definition of
        Eidenbenz, Locher -- Task Allocation for Distributed Stream Processing (Extended Version)

        :param n:
        :return:
        """
        if n == 0:
            return nx.DiGraph()
        elif n == 1:
            G = nx.DiGraph()
            # all nodes are unique
            G.add_node(self.current_node_id)
            self.current_node_id += 1
            return G
        else:
            n1 = int(math.floor(n * self.range_splitter))
            n2 = n - n1
            G1 = self.series_parallel_decomposable_generator(n1)
            G2 = self.series_parallel_decomposable_generator(n2)
            if self.parallel_serial_ratio < self.random.random():
                # parallel composition, creates a new graph without adding edges
                return nx.compose(G1, G2)
            else:
                # series composition
                G = nx.DiGraph()
                sources_of_G2 = []
                for n, d in G2.in_degree:
                    if d == 0:
                        sources_of_G2.append(n)
                # iterate on the sinks of G1
                for n, d in G1.out_degree:
                    if d == 0:
                        for source in sources_of_G2:
                            G.add_edge(n, source)
                return nx.compose(nx.compose(G, G1), G2)

    def series_parallel_generator(self, n):
        """
        Creates SP graphs with this definition: http://www.graphclasses.org/classes/gc_275.html

        :param n:
        :return:
        """
        G = nx.MultiDiGraph()
        G.add_node(self.current_node_id)
        G.add_edge(self.current_node_id, self.current_node_id)
        self.current_node_id += 1
        while G.number_of_nodes() < n:
            p = self.random.random()
            u, v = self.random.choice(list(G.edges()))
            if p < self.range_splitter:
                # subdivide an edge
                # deterministically remove always the smallest ID edge
                k = min(G[u][v].keys())
                G.remove_edge(u, v, k)
                G.add_node(self.current_node_id)
                G.add_edge(u, self.current_node_id)
                G.add_edge(self.current_node_id, v)
                self.current_node_id += 1
            else:
                # add parallel edge
                G.add_edge(u, v)
        return G

    def _allowed_substrate_node(self, demand, substrate):
        """
        Choses a substrate node which has enough resources.

        :param demand:
        :param substrate:
        :return: list of a single node id
        """
        substrate_node_tuples = [t for t in substrate.node.iteritems()]
        self.random.shuffle(substrate_node_tuples)
        for substrate_node_id, d in substrate_node_tuples:
            if d['capacity'][self.universal_node_type] > demand and \
                    substrate_node_id not in self.substrate_nodes_with_bounded_app_node:
                self.substrate_nodes_with_bounded_app_node.add(substrate_node_id)
                self.logger.debug("Choosing substrate node {} to host a location bound request node with demand {}".format(
                                                                                                                    substrate_node_id, demand))
                return [substrate_node_id]
        # this has a quite low probablility due to the high node capacity / demand ration
        self.logger.warn("Capacity demand {} cannot be found on any non-already app node location bound substrate nodes, scenario cannot "
                         "be feasible!".format(demand))
        return None

    def convert_nx_to_alib_graph(self, name, G_nx, location_bound_node_ids, substrate):
        """
        Creates a datamodel.Request object based on a network X style graph

        :param name:
        :param G_nx:
        :param location_bound_node_ids:
        :param substrate:
        :return:
        """
        req = datamodel.Request(name)
        for node_id in G_nx.nodes:
            capacity_demand = self.random.uniform(self.min_node_demand, self.max_node_demand)
            allowed_node_list = None
            if node_id in location_bound_node_ids:
                allowed_node_list = self._allowed_substrate_node(capacity_demand, substrate)
            req.add_node(node_id, demand=capacity_demand, ntype=self.universal_node_type,
                         allowed_nodes=allowed_node_list)
        for edge in G_nx.edges:
            tail = edge[0]
            head = edge[1]
            # if there are parallel edges or loops in the generated NX graph we discard
            if (tail, head) not in req.edges and tail != head:
                # NOTE: edge might contain 2 or 3 elements, 3rd being the edge key in multidigraph.
                capacity_demand = self.random.uniform(self.min_link_demand, self.max_link_demand)
                # directed request edge by default
                req.add_edge(tail, head, demand=capacity_demand)
        return req

    def generate_request(self, name, raw_parameters, substrate):
        # NOTE: maybe should it return just the list in order? (to comply with the framework), but this function is not called directly
        # anywhere
        raise NotImplementedError("This generator creates the request list at once!")

    def generate_request_list(self, raw_parameters, substrate, base_name="vnet_{id}", normalize=False):
        """
        Realizes the generator function to fit to the framework.

        :param raw_parameters:
        :param substrate:
        :return:
        """
        self._read_raw_parameters(raw_parameters, substrate)
        self.logger.info("Generating series parallel decomposable request graph with {} nodes".format(self.node_count))
        req_list = []

        if self.use_connected_sp_def:
            graph_components_iterable = []
            total_node_count = 0
            single_req_size = int(math.ceil(self.node_count / float(self.number_of_requests)))
            self.logger.info("Generating requests of size {}".format(single_req_size))
            while total_node_count < self.node_count:
                single_component_node_cnt = min(single_req_size,
                                                self.node_count - total_node_count)
                total_node_count += single_component_node_cnt
                G_nx = self.series_parallel_generator(single_component_node_cnt)
                if len(list(nx.weakly_connected_component_subgraphs(G_nx))) > 1:
                    raise sg.RequestGenerationError("Series parallel graph is not connected!")
                graph_components_iterable.append(G_nx)
        else:
            G_nx = self.series_parallel_decomposable_generator(self.node_count)
            graph_components_iterable = nx.weakly_connected_component_subgraphs(G_nx)

        # must be common for all connected components
        location_bound_node_ids = self.random.sample(list(range(0, self.current_node_id)),
                                                     int(self.node_count * self.location_bound_mapping_ratio))
        self.logger.debug("Choosing location bound request nodes: {}".format(location_bound_node_ids))

        req_graph_count = 1
        for G_nx in graph_components_iterable:
            name = "fog_app_" + base_name.format(id = req_graph_count)
            req = self.convert_nx_to_alib_graph(name, G_nx, location_bound_node_ids, substrate)
            req_graph_count += 1
            self.logger.debug("Adding request graph on edges {} as a separate request".format(set(G_nx.edges())))
            req_list.append(req)

        if len(req_list) > self.number_of_requests:
            self.logger.warn("More requests are generated than the required number of requests, discarding the "
                             "excess graphs...")
        if normalize:
            self.normalize_resource_footprint(raw_parameters, req_list, substrate)

        return req_list[:self.number_of_requests]

