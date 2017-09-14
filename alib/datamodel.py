# MIT License
#
# Copyright (c) 2016-2017 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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

class SubstrateError(Exception): pass


class LinearRequestError(Exception): pass


class Objective(object):
    MIN_COST = 0
    MAX_PROFIT = 1


class Scenario(object):
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


class Graph(object):
    """ representing a directed graph ( G = ( V , E) )
        Attributes :
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
        self.shortest_paths_costs = {}
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
    """ represents a request graph ?
    """

    def __init__(self, name):
        super(Request, self).__init__(name)
        self.graph['latency_requirement'] = {}
        self.types = set()
        self.profit = 0.0

    def add_node(self, i, demand, ntype, allowed_nodes=None):
        super(Request, self).add_node(i, demand=demand, allowed_nodes=allowed_nodes, type=ntype)
        self.types.add(ntype)

    def add_edge(self, tail, head, demand, allowed_edges=None):
        if tail in self.nodes and head in self.nodes:
            super(Request, self).add_edge(tail, head, demand=demand, allowed_edges=allowed_edges)

    def add_latency_requirement(self, path, latency):
        """ adds to a specific 'path' a latency requirement
            important: the order of edges must be respected """
        if set(path) <= self.edges:
            self.graph['latency_requirement'][tuple(path)] = latency
        else:
            print "Path contains edges which are NOT in request edges"

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

    def get_latency_requirement(self, path):
        return self.graph['latency_requirement'][tuple(path)]

    def get_all_latency_requirements(self):
        return self.graph['latency_requirement']

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
    """ represents a linear request / service chain graph
    """

    def __init__(self, name):
        super(LinearRequest, self).__init__(name)
        self.graph['latency_requirement'] = {}
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

    def add_latency_requirement(self, path, latency):
        """ adds to a specific 'path' a latency requirement
            important: the order of edges must be respected """
        if set(path) <= self.edges:
            self.graph['latency_requirement'][tuple(path)] = latency
        else:
            print "Path contains edges which are NOT in request edges"

    def get_out_edge(self, i):
        if len(self.out_edges[i]) == 0:
            return None
        elif len(self.out_edges[i]) == 1:
            return self.out_edges[i][0]
        else:
            raise LinearRequestError("Linear Request cannot have multiple outgoing edges!")

    def get_latency_requirement(self, path):
        return self.graph['latency_requirement'][tuple(path)]

    def get_all_latency_requirements(self):
        return self.graph['latency_requirement']

    def get_required_types(self):
        return self.types


class Substrate(Graph):
    """ representing a single substrate ( G_s = ( V_s , E_s) )
        Attributes :
    """

    def __init__(self, name):
        super(Substrate, self).__init__(name)
        self.types = set()

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

    def add_edge(self, tail, head, latency=1.0, capacity=1.0, cost=1.0, bidirected=True):
        if tail in self.nodes and head in self.nodes:
            # is always bidirected
            super(Substrate, self).add_edge(tail, head, bidirected=bidirected,
                                            latency=latency,
                                            capacity=capacity,
                                            cost=cost)
        else:
            raise SubstrateError("Nodes {} and/or {} are not in the substrate:\n{}".format(tail, head, self.nodes))

    def get_types(self):
        return self.types

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
        return self.node[node]['capacity']

    def get_node_type_capacity(self, node, ntype):
        return self.node[node]['capacity'].get(ntype, 0.0)

    def average_node_capacity(self, node_type):
        return self.get_total_node_resources(node_type) / float(len(self.get_nodes_by_type(node_type)))

    def get_path_latency(self, path):
        return sum(map(self.get_edge_latency, path))

    def get_path_capacity(self, path):
        return min(map(self.get_edge_capacity, path))

    def get_edge_latency(self, edge):
        return self.edge[edge]['latency']

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
