# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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

from collections import defaultdict
import random
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


class SubstrateError(Exception): pass


class LinearRequestError(Exception): pass


class Objective(object):
    ''' Enum representing the objective that shall be applied.

    '''
    MIN_COST = 0
    MAX_PROFIT = 1


class Scenario(object):
    ''' Represents the scenario of embedding a given number of requests onto a single substrate network.

    '''
    def __init__(self, name, substrate, requests, objective=Objective.MIN_COST):
        self.name = name
        self.requests = requests
        self.substrate = substrate
        self.objective = objective

    def get_substrate(self):
        return self.substrate

    def get_requests(self):
        return self.requests

    def validate_types(self):
        """ general check for every request if substrate supports every needed
        type
        """
        for req in self.requests:
            required_types = req.get_required_types()
            available_types = self.substrate.get_types()
            if not (required_types <= available_types):
                print required_types - available_types, ' missing'
                return False
        return True


class UndirectedGraph(object):
    """ Simple representation of an unidrected graph (without any further attributes as weights, costs etc.)
    """
    def __init__(self, name):
        self.name = name
        self.nodes = set()
        self.edges = set()

        self.neighbors = {}
        self.incident_edges = {}

    def add_node(self, node):
        self.nodes.add(node)
        self.neighbors[node] = set()
        self.incident_edges[node] = set()

    def add_edge(self, i, j):
        if i not in self.nodes or j not in self.nodes:
            raise ValueError("Nodes not in graph!")
        new_edge = frozenset([i, j])
        if new_edge in self.edges:
            raise ValueError("Duplicate edge {new_edge}!")
        if len(new_edge) == 1:
            raise ValueError("Loop edges are not allowed ({i})")

        self.neighbors[i].add(j)
        self.neighbors[j].add(i)
        self.incident_edges[i].add(new_edge)
        self.incident_edges[j].add(new_edge)
        self.edges.add(new_edge)
        return new_edge

    def remove_node(self, node):
        if node not in self.nodes:
            raise ValueError("Node not in graph.")

        edges_to_remove = list(self.incident_edges[node])

        for incident_edge in edges_to_remove:
            edge_as_list = list(incident_edge)
            self.remove_edge(edge_as_list[0], edge_as_list[1])

        del self.incident_edges[node]
        del self.neighbors[node]
        self.nodes.remove(node)

    def remove_edge(self, i, j):
        old_edge = frozenset([i, j])
        if i not in self.nodes or j not in self.nodes:
            raise ValueError("Nodes not in graph!")
        if old_edge not in self.edges:
            raise ValueError("Edge not in graph!")

        self.neighbors[i].remove(j)
        self.neighbors[j].remove(i)

        self.incident_edges[i].remove(old_edge)
        self.incident_edges[j].remove(old_edge)

        self.edges.remove(old_edge)

    def get_incident_edges(self, node):
        return self.incident_edges[node]

    def get_neighbors(self, node):
        return self.neighbors[node]

    def get_edge_representation(self):
        return [list(edge) for edge in self.edges]

    def check_connectedness(self):
        if len(self.nodes) == 0:
            return True
        root = next(iter(self.nodes))
        unvisited = set(self.nodes)
        to_process = [root]
        while len(to_process) > 0:
            current_node = to_process.pop(0)
            if not current_node in unvisited:
                continue
            for neighbor in self.neighbors[current_node]:
                if neighbor in unvisited:
                    to_process.append(neighbor)
            unvisited.remove(current_node)
        return len(unvisited) == 0





    def __str__(self):
        return "{} {} with following attributes: \n\t\tNodes{}\n\t\tEdges{}".format(type(self).__name__, self.name,
                                                                                    self.nodes, self.edges)

def get_undirected_graph_from_edge_representation(edge_list, name=""):
    ''' returns an undirected graph given a list of edges
    '''

    graph = UndirectedGraph(name=name)

    for i,j in edge_list:
        if i not in graph.nodes:
            graph.add_node(i)
        if j not in graph.nodes:
            graph.add_node(j)
        graph.add_edge(i,j)
    
    return graph

def is_connected_undirected_edge_representation(edge_list):
    ''' Given a list of edges, returns whether the result undirected graph is connected
    '''
    #each node is assigned the connected component id
    node_to_connected_component_id = {}
    # holds the nodes for each connected component ids
    connected_component_id_to_nodes = defaultdict(list)

    new_connected_component_id = 0

    for i,j in edge_list:

        if i in node_to_connected_component_id.keys() and j in node_to_connected_component_id.keys():
            #both nodes are already known
            connected_component_i = node_to_connected_component_id[i]
            connected_component_j = node_to_connected_component_id[j]

            if connected_component_i != connected_component_j:
                #merge connected components, removing the connected component of node j

                for k in connected_component_id_to_nodes[connected_component_j]:
                    node_to_connected_component_id[k] = connected_component_i

                connected_component_id_to_nodes[connected_component_i].extend(connected_component_id_to_nodes[connected_component_j])
                del connected_component_id_to_nodes[connected_component_j]

        elif i in node_to_connected_component_id.keys() and j not in node_to_connected_component_id.keys():
            #add j to connected component of i
            connected_component_i = node_to_connected_component_id[i]
            node_to_connected_component_id[j] = connected_component_i
            connected_component_id_to_nodes[connected_component_i].append(j)

        elif i not in node_to_connected_component_id.keys() and j in node_to_connected_component_id.keys():
            # add i to connected component of j
            connected_component_j = node_to_connected_component_id[j]
            node_to_connected_component_id[i] = connected_component_j
            connected_component_id_to_nodes[connected_component_j].append(i)

        else:
            # create new connected component and add both nodes
            new_connected_component_id += 1
            node_to_connected_component_id[i] = new_connected_component_id
            node_to_connected_component_id[j] = new_connected_component_id
            connected_component_id_to_nodes[new_connected_component_id].extend([i,j])

    if len(connected_component_id_to_nodes.keys()) == 1:
        return True
    else:
        return False


def get_nodes_of_edge_list_representation(undirected_graph_edge_representation):
    nodes = set()
    for i, j in undirected_graph_edge_representation:
        nodes.add(i)
        nodes.add(j)
    return list(nodes)

def get_number_of_nodes_edge_list_representation(undirected_graph_edge_representation):
    return len(get_nodes_of_edge_list_representation(undirected_graph_edge_representation))



class UndirectedGraphStorage(object):

    def __init__(self, parameter_name, random_instance=None):
        self.parameter_name = parameter_name
        self.undirected_edge_representation_storage = {}
        if random_instance is None:
            random_instance = random.Random()
        self.random_instance = random_instance
        self._average_number_of_edges_dict = {}

    def add_graph_as_edge_representation(self, parameter, edge_representation):
        if parameter not in self.undirected_edge_representation_storage:
            self.undirected_edge_representation_storage[parameter] = {}
        number_of_nodes = get_number_of_nodes_edge_list_representation(edge_representation)
        if number_of_nodes not in self.undirected_edge_representation_storage[parameter]:
            self.undirected_edge_representation_storage[parameter][number_of_nodes] = []

        self.undirected_edge_representation_storage[parameter][number_of_nodes].append(edge_representation)

    def load_from_pickle(self, pickle_path):
        other_undirected_graph_storage = None
        with open(pickle_path, "r") as f:
            other_undirected_graph_storage = pickle.load(f)

        self.parameter_name = other_undirected_graph_storage.parameter_name
        self.undirected_edge_representation_storage = other_undirected_graph_storage.undirected_edge_representation_storage

    def add_graph(self, parameter, graph):
        self.add_graph_as_edge_representation(parameter, graph.get_edge_representation())


    def get_random_graph(self, parameter, number_of_nodes, name=""):
        edge_graph_representation = self.get_random_graph_as_edge_list_representation(parameter, number_of_nodes)
        return get_undirected_graph_from_edge_representation(edge_graph_representation, name)

    def get_random_graph_as_edge_list_representation(self, parameter, number_of_nodes):
        if parameter not in self.undirected_edge_representation_storage:
            raise ValueError("No graphs are stored for parameter {}".format(parameter))
        if number_of_nodes not in self.undirected_edge_representation_storage[parameter]:
            raise ValueError("No graphs are stored for parameter {} and number of nodes {}".format(parameter, number_of_nodes))

        number_of_potential_graphs = len(self.undirected_edge_representation_storage[parameter][number_of_nodes])
        selected_index = self.random_instance.randint(0, number_of_potential_graphs-1)
        return self.undirected_edge_representation_storage[parameter][number_of_nodes][selected_index]

    def _get_edge_distribution_information(self, parameter_value, number_of_nodes):
        number_of_graphs = len(self.undirected_edge_representation_storage[parameter_value][number_of_nodes])
        edge_counts = np.zeros(number_of_graphs)
        for i in range(number_of_graphs):
            edge_counts[i] = len(self.undirected_edge_representation_storage[parameter_value][number_of_nodes][i])
        sorted_edge_counts = np.sort(edge_counts)
        interesting_indices = [0] + [int(number_of_graphs*i/100.0) for i in [5,25,50,75,95]] + [number_of_graphs-1]
        percentiles = [sorted_edge_counts[index] for index in interesting_indices]
        return "min: {:>2}, max: {:>2}, median: {:>2};   5, 25, 75, 95 percentiles: {:>2} {:>2} {:>2} {:>2}".format(percentiles[0],
                                                                                                                    percentiles[6],
                                                                                                                    percentiles[3],
                                                                                                                    percentiles[1],
                                                                                                                    percentiles[2],
                                                                                                                    percentiles[4],
                                                                                                                    percentiles[5])

    def get_average_number_of_edges_for_parameter(self, parameter_value):
        if parameter_value in self._average_number_of_edges_dict.keys():
            return self._average_number_of_edges_dict[parameter_value]
        else:
            self._average_number_of_edges_dict[parameter_value] = {}
            for foolaa in self.undirected_edge_representation_storage[parameter_value].keys():
                number_of_graphs = len(self.undirected_edge_representation_storage[parameter_value][foolaa])
                edge_counts = np.zeros(number_of_graphs)
                for i in range(number_of_graphs):
                    edge_counts[i] = len(
                        self.undirected_edge_representation_storage[parameter_value][foolaa][i])
                self._average_number_of_edges_dict[parameter_value][foolaa] = np.average(edge_counts)
            return self._average_number_of_edges_dict[parameter_value]


    def get_information(self):
        result = ""
        for parameter_value in self.undirected_edge_representation_storage.keys():
            total = 0
            result += "========================\nPARAMETER {} = {}\n========================\n".format(self.parameter_name, parameter_value)
            for number_of_nodes in self.undirected_edge_representation_storage[parameter_value].keys():
                number_of_graphs = len(self.undirected_edge_representation_storage[parameter_value][number_of_nodes])
                edge_info = self._get_edge_distribution_information(parameter_value, number_of_nodes)
                result += "\tnodes: {:>3} --> {:>5} graphs with edge distribution {}\n".format(number_of_nodes, number_of_graphs, edge_info)
                total += number_of_graphs
            result += "========================\nTOTAL: {} graphs\n========================\n".format(total)
        return result


class Graph(object):
    """ Representing a directed graph ( G = ( V , E) ).

        Arbitrary attributes can be set for nodes and edges via **kwargs.
    """

    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.nodes = set()
        self.edges = set()
        self.out_neighbors = {}
        self.in_neighbors = {}
        self.out_edges = {}
        self.in_edges = {}
        self.node = {}
        self.edge = {}
        self.shortest_paths_costs = None
        self._shortest_paths_attribute_identifier = "cost"

    def add_node(self, node, **kwargs):

        self.nodes.add(node)
        self.out_neighbors[node] = []
        self.in_neighbors[node] = []
        self.out_edges[node] = []
        self.in_edges[node] = []
        self.node[node] = {}
        for key, value in kwargs.items():
            self.node[node][key] = value
        #print "added node: {}, now there are {} nodes".format(node, len(self.nodes))

    def add_edge(self, tail, head, bidirected=False, **kwargs):
        if (tail not in self.nodes) or (head not in self.nodes):
            raise Exception("Node was not found while adding edge")

        self._add_edge_one_direction(tail=tail, head=head, **kwargs)
        if bidirected:
            self._add_edge_one_direction(tail=head, head=tail, **kwargs)

    def _add_edge_one_direction(self, tail, head, **kwargs):
        new_edge = (tail, head)
        if new_edge in self.edges:
            raise ValueError("Duplicate edge {}!".format(new_edge))
        if tail == head:
            raise ValueError("Loop edges are not allowed ({0})".format(tail))

        self.out_neighbors[tail].append(head)
        self.in_neighbors[head].append(tail)
        self.out_edges[tail].append(new_edge)
        self.in_edges[head].append(new_edge)

        self.edges.add(new_edge)
        self.edge[new_edge] = {}
        for key, value in kwargs.items():
            self.edge[new_edge][key] = value

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_out_neighbors(self, node):
        return self.out_neighbors[node]

    def get_in_neighbors(self, node):
        return self.in_neighbors[node]

    def get_out_edges(self, node):
        return self.out_edges[node]

    def get_in_edges(self, node):
        return self.in_edges[node]

    def get_name(self):
        return self.name

    def get_number_of_nodes(self):
        return len(self.nodes)

    def get_number_of_edges(self):
        return len(self.edges)

    def get_shortest_paths_cost(self, node, other):
        if self.shortest_paths_costs is None:
            self.initialize_shortest_paths_costs()
        return self.shortest_paths_costs[node][other]

    def get_shortest_paths_cost_dict(self):
        if self.shortest_paths_costs is None:
            self.initialize_shortest_paths_costs()
        return self.shortest_paths_costs

    def initialize_shortest_paths_costs(self):
        # this can only be used if costs are defined as such for each edge

        self.shortest_paths_costs = {}

        for edge in self.edges:
            if self._shortest_paths_attribute_identifier not in self.edge[edge]:
                raise Exception("cost not defined for edge {}".format(edge))

        for u in self.nodes:
            self.shortest_paths_costs[u] = {}
            for v in self.nodes:
                if u is v:
                    self.shortest_paths_costs[u][v] = 0
                else:
                    self.shortest_paths_costs[u][v] = None

        for (u, v) in self.edges:
            self.shortest_paths_costs[u][v] = self.edge[(u, v)][self._shortest_paths_attribute_identifier]

        for k in self.nodes:
            for u in self.nodes:
                for v in self.nodes:
                    if self.shortest_paths_costs[u][k] is not None and self.shortest_paths_costs[k][v] is not None:
                        cost_via_k = self.shortest_paths_costs[u][k] + self.shortest_paths_costs[k][v]
                        if self.shortest_paths_costs[u][v] is None or cost_via_k < self.shortest_paths_costs[u][v]:
                            self.shortest_paths_costs[u][v] = cost_via_k

    def check_connectivity(self):
        if self.shortest_paths_costs is None:
            self.initialize_shortest_paths_costs()
        for u in self.nodes:
            for v in self.nodes:
                if self.shortest_paths_costs[u][v] is None:
                    print "node {} cannot reach node {}".format(u, v)
                    return False


        return True

    def __str__(self):
        return "{} {} with following attributes: \n\t\tNodes{}\n\t\tEdges{}".format(type(self).__name__, self.name,
                                                                                    self.nodes, self.edges)

    def print_shortest_path(self, including_shortest_path_costs=True, data=False):
        if data:
            print self.__str__()
        if including_shortest_path_costs:
            if self.shortest_paths_costs is None:
                self.initialize_shortest_paths_costs()
            print "Distances:"
            for u in self.nodes:
                for v in self.nodes:
                    print "{} to {}: {}".format(u, v, self.shortest_paths_costs[u][v])


class Request(Graph):
    """ Represents a request graph, i.e. a directed graph with demands for nodes and edges.

        Note that each node is attributed with a resource type.
    """

    def __init__(self, name):
        super(Request, self).__init__(name)
        self.types = set()
        self.profit = 0.0

    def add_node(self, i, demand, ntype, allowed_nodes=None):
        super(Request, self).add_node(i, demand=demand, allowed_nodes=allowed_nodes, type=ntype)
        self.types.add(ntype)

    def add_edge(self, tail, head, demand, allowed_edges=None):
        super(Request, self).add_edge(tail, head, demand=demand, allowed_edges=allowed_edges)

    def set_allowed_nodes(self, i, allowed_nodes):
        if i in self.nodes:
            self.node[i]['allowed_nodes'] = allowed_nodes
        else:
            print "Request nodes are NOT contained in request"

    def get_allowed_nodes(self, i):
        return self.node[i]['allowed_nodes']

    def get_allowed_edges(self, ij):
        if ij not in self.edges:
            print "Edge {} is NOT contained in request".format(ij)
        elif "allowed_edges" in self.edge[ij]:
            return self.edge[ij]['allowed_edges']
        return None

    def set_allowed_edges(self, ij, allowed_edges):
        if ij in self.edges:
            self.edge[ij]['allowed_edges'] = allowed_edges
        else:
            print "Request edge {} is NOT contained in request".format(ij)

    def get_required_types(self):
        return self.types

    def get_node_demand(self, i):
        if i in self.node:
            return self.node[i]["demand"]
        else:
            print "Node {} is NOT contained in request".format(i)

    def get_type(self, i):
        if i in self.node:
            return self.node[i]["type"]
        else:
            print "Node {} is NOT contained in request".format(i)

    def get_edge_demand(self, ij):
        if ij in self.edge:
            return self.edge[ij]["demand"]
        else:
            print "Edge {} is NOT contained in request".format(ij)

    def get_nodes_by_type(self, nt):
        return [i for i in self.nodes if self.get_type(i) == nt]

    def __str__(self):
        return super(Request, self).__str__()


class LinearRequest(Request):
    """ Represents a linear request, i.e. a service chain graph: virtual nodes are chained.
    """

    def __init__(self, name):
        super(LinearRequest, self).__init__(name)
        self.sequence = []
        self.types = set()

    # ASSUMPTION: nodes are added in order of their actual service chain order
    def add_node(self, i, demand, ntype, allowed_nodes=None):
        super(LinearRequest, self).add_node(i,
                                            demand=demand,
                                            allowed_nodes=allowed_nodes,
                                            ntype=ntype)
        self.sequence.append(i)
        self.types.add(ntype)

    def add_edge(self, tail, head, demand):
        if len(self.out_edges[tail]) != 0:
            raise LinearRequestError("Linear Request cannot have multiple outgoing edges!")
        if tail in self.nodes and head in self.nodes:
            super(LinearRequest, self).add_edge(tail, head, demand=demand)

    def get_out_edge(self, i):
        if len(self.out_edges[i]) == 0:
            return None
        elif len(self.out_edges[i]) == 1:
            return self.out_edges[i][0]
        else:
            raise LinearRequestError("Linear Request cannot have multiple outgoing edges!")

    def get_required_types(self):
        return self.types


class Substrate(Graph):
    """ Represents a physical network.

        The constructor must be passed a set of network function types. Each substrate node may host an arbitrary subset
        of these functions and can have arbitrary capacities for each of these.
    """

    def __init__(self, name):
        super(Substrate, self).__init__(name)
        self.types = set()
        self._average_node_distance = None

    def add_node(self, u, types, capacity, cost):
        super(Substrate, self).add_node(u, supported_types=types,
                                        capacity=capacity,
                                        cost=cost)
        if isinstance(types, basestring):
            raise SubstrateError("Types should be a list or set of strings, not a single string. ({})".format(types))
        for node_type in types:
            if node_type not in capacity:
                raise SubstrateError("No capacity defined for type {}".format(node_type))
            if node_type in self.nodes:
                raise SubstrateError("Type {} is also a node in the substrate.".format(node_type))
            self.types.add(node_type)

    def add_edge(self, tail, head, capacity=1.0, cost=1.0, bidirected=True, **kwargs):
        if tail in self.nodes and head in self.nodes:
            # is always bidirected
            super(Substrate, self).add_edge(tail, head, bidirected=bidirected,
                                            capacity=capacity,
                                            cost=cost, **kwargs)
        else:
            raise SubstrateError("Nodes {} and/or {} are not in the substrate:\n{}".format(tail, head, self.nodes))

    def set_average_node_distance(self, dist):
        self._average_node_distance = dist

    def get_average_node_distance(self):
        return self._average_node_distance

    def get_types(self):
        return self.types

    def get_supported_node_types(self, node):
        return self.node[node]['supported_types']

    def get_nodes_by_type(self, ntype):
        nodes = []
        for u in self.nodes:
            if ntype in self.node[u]['supported_types']:
                nodes.append(u)
        return nodes

    def get_node_cost(self, node):
        return self.node[node]['cost']

    def get_node_type_cost(self, node, ntype):
        return self.node[node]['cost'][ntype]

    def get_node_capacity(self, node):
        if isinstance(self.node[node]["capacity"], float):
            return self.node[node]["capacity"]
        else:
            if len(self.node[node]["capacity"]) > 1:
                raise RuntimeError("Type has to be specified when a node hosts more than one type.")
            else:
                return next(iter(self.node[node]['capacity'].values()))

    def get_node_type_capacity(self, node, ntype):
        return self.node[node]['capacity'].get(ntype, 0.0)

    def average_node_capacity(self, node_type):
        return self.get_total_node_resources(node_type) / float(len(self.get_nodes_by_type(node_type)))

    def get_path_capacity(self, path):
        return min(map(self.get_edge_capacity, path))

    def get_edge_cost(self, edge):
        return self.edge[edge]['cost']

    def get_edge_capacity(self, edge):
        return self.edge[edge]['capacity']

    def average_edge_capacity(self):
        return sum(self.get_edge_capacity(e) for e in self.edges) / float(len(self.edges))

    def get_total_edge_resources(self):
        return sum(self.get_edge_capacity(e) for e in self.edges)

    def get_total_node_resources(self, node_type):
        return sum(self.get_node_type_capacity(u, node_type) for u in self.nodes)

    def __str__(self):
        return super(Substrate, self).__str__()


class SubstrateX(object):
    """
    Extends the substrate class with efficient lookup of substrate resources given a capacity requirement.

    This is separated from the Substrate class so that the resource data is not included in pickle files.
    """

    def __init__(self, substrate):
        self.substrate = substrate
        self.substrate_edge_resources = sorted(self.substrate.edges,
                                               key=lambda edge: self.substrate.edge[edge]['capacity'])
        self.substrate_node_resources = []
        self.substrate_resources = list(self.substrate_edge_resources)
        self.substrate_resource_capacities = {}
        self._demand_mapped_index = {}
        self._list_of_edge_resource_sets = []
        self._list_of_edge_resource_caps = []

        for sedge in self.substrate.edges:
            self.substrate_resource_capacities[sedge] = self.substrate.edge[sedge]['capacity']

        for ntype in self.substrate.get_types():
            for snode in self.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
                self.substrate_resources.append((ntype, snode))
                #print self.substrate.node[snode]
                self.substrate_resource_capacities[(ntype, snode)] = self.substrate.node[snode]['capacity'][ntype]

        self.substrate_node_resources = sorted(
            self.substrate_node_resources,
            key=lambda node_res: self.substrate_resource_capacities[node_res]
        )

        self._list_of_node_resource_sets = {nt: [] for nt in self.substrate.get_types()}
        self._list_of_node_resource_caps = {nt: [] for nt in self.substrate.get_types()}

        self._initialize_edge_demand_lookup()
        self._initialize_node_demand_lookup()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError("Substrate has no attribute '%s'" % name)
        try:
            return getattr(self.substrate, name)
        except AttributeError as e:
            raise AttributeError("Substrate has no attribute '%s'" % name)

    def _initialize_edge_demand_lookup(self):
        current_cap = None
        _list_of_edge_resource_sets = []
        for uv in self.substrate_edge_resources:
            cap = self.substrate.edge[uv]["capacity"]
            if cap != current_cap:
                _list_of_edge_resource_sets.append(set())
                self._list_of_edge_resource_caps.append(cap)
                current_cap = cap
            for _set in _list_of_edge_resource_sets:
                _set.add(uv)
        self._list_of_edge_resource_caps.append(float("inf"))
        _list_of_edge_resource_sets.append(set())
        self._list_of_edge_resource_sets = [frozenset(x) for x in _list_of_edge_resource_sets]

    def _initialize_node_demand_lookup(self):
        current_cap = {nt: -1 for nt in self.substrate.get_types()}
        for t, u in self.substrate_node_resources:
            cap = self.substrate.get_node_type_capacity(u, t)
            if cap != current_cap[t]:
                self._list_of_node_resource_sets[t].append(set())
                self._list_of_node_resource_caps[t].append(cap)
                current_cap[t] = cap
            for _set in self._list_of_node_resource_sets[t]:
                _set.add(u)
        for t in self.substrate.get_types():
            self._list_of_node_resource_caps[t].append(float("inf"))
            self._list_of_node_resource_sets[t].append(set())

        self._list_of_node_resource_sets = {
            t: [frozenset(x) for x in self._list_of_node_resource_sets[t]]
            for t in self._list_of_node_resource_sets
        }

    def get_valid_edges(self, demand):
        if demand in self._demand_mapped_index:
            return self._list_of_edge_resource_sets[self._demand_mapped_index[demand]]
        low_index = 0
        high_index = len(self._list_of_edge_resource_caps) - 1
        res_index = None
        while low_index < high_index:
            index = (low_index + high_index) / 2
            cap = self._list_of_edge_resource_caps[index]
            if demand > cap:
                low_index = index + 1
            elif demand < cap:
                high_index = index - 1
            else:
                res_index = index
                break
            res_index = low_index
        if self._list_of_edge_resource_caps[res_index] < demand:
            res_index += 1
        result = self._list_of_edge_resource_sets[res_index]
        self._demand_mapped_index[demand] = res_index
        return result

    def get_valid_nodes(self, type, demand):
        caps = self._list_of_node_resource_caps[type]
        sets = self._list_of_node_resource_sets[type]
        if (type, demand) in self._demand_mapped_index:
            return sets[self._demand_mapped_index[(type, demand)]]

        low_index = 0
        high_index = len(caps) - 1
        res_index = None
        while low_index < high_index:
            index = (low_index + high_index) / 2
            cap = caps[index]
            if demand > cap:
                low_index = index + 1
            elif demand < cap:
                high_index = index - 1
            else:
                res_index = index
                break
            res_index = low_index
        if caps[res_index] < demand:
            res_index += 1
        result = sets[res_index]
        self._demand_mapped_index[(type, demand)] = res_index
        return result
