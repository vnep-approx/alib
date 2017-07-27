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

from . import util

log = util.get_logger(__name__, make_file=False, propagate=True)


class SolutionStorageError(Exception):    pass


class MappingError(Exception): pass


class IntegralScenarioSolution(object):
    def __init__(self, name, scenario):
        self.name = name
        self.scenario = scenario
        # this a dictionary of request --> mapping
        self.request_mapping = {}

    def add_mapping(self, req, mapping):
        self.request_mapping[req] = mapping

    def validate_solution(self):
        """ validates types, latency and capacity for each request and given
        mapping in scenario
        """
        if (not self.scenario.validate_types()):
            return False
        for request in self.scenario.requests:
            if (request in self.request_mapping):
                mapping = self.request_mapping[request]
                substrate = self.scenario.substrate
                rules = [self.type_check(request, mapping, substrate),
                         self.latency_check(request, mapping, substrate),
                         self.capacity_check(request, mapping, substrate)]
                if not all(rules):
                    return False
        return True

    def validate_solution_fulfills_capacity(self):
        substrate = self.scenario.substrate
        substrate_resources = {uv: substrate.edge[uv]["capacity"] for uv in substrate.edges}
        for u in substrate.nodes:
            for ntype in substrate.node[u]["supported_types"]:
                substrate_resources[ntype, u] = substrate.node[u]["capacity"][ntype]
        for req, mapping in self.request_mapping.items():
            for i, u in mapping.mapping_nodes.items():
                t = req.node[i]["type"]
                demand = req.node[i]["demand"]
                substrate_resources[t, u] -= demand
            for ij, uv_list in mapping.mapping_edges.items():
                demand = req.edge[ij]["demand"]
                for uv in uv_list:
                    substrate_resources[uv] -= demand
        for res, remaining_cap in substrate_resources.items():
            if remaining_cap < 0:
                print res, "violated capacity by ", -remaining_cap

    def type_check(self, request, mapping, substrate):
        """ checks if requested types are in supported types from substrate
        """
        for i in mapping.mapping_nodes:
            u = mapping.mapping_nodes[i]
            if (request.node[i]['type'] not in
                    substrate.node[u]['supported_types']):
                print "Node:", u, " does not support type:", request.node[i]['type']
                return False
        return True

    def latency_check(self, request, mapping, substrate):
        for path in mapping.request.graph['latency_requirement'].keys():
            sum = 0
            for ve in path:
                sum += substrate.get_path_latency(mapping.mapping_edges[ve])
                if sum > request.graph['latency_requirement'][path]:
                    return False
        return True

    def capacity_check(self, request, mapping, substrate):
        """ checks if demand of all request nodes and edges is fullfilled by
        substrate capacity
        """
        for i in mapping.mapping_nodes:
            if request.node[i]['demand'] > substrate.node[mapping.mapping_nodes[i]]['capacity']:
                return False
        for ve in mapping.request.edge:
            if request.edge[ve]['demand'] > substrate.get_path_capacity(mapping.mapping_edges[ve]):
                return False
        return True

    def __str__(self):
        s = ""
        for req in self.scenario.requests:
            s += "\t" + str(req) + "\n"
            if req in self.request_mapping.keys():
                s += "\t" + str(self.request_mapping[req]) + "\n"
            else:
                s += "\tnot embedded no mapping \n"
        return "ScenarioSolution {} for:\n{}".format(self.name, s)


class FractionalScenarioSolution(object):
    def __init__(self, name, scenario):
        self.name = name
        self.scenario = scenario
        # this a dictionary of request --> list of mappings
        self.request_mapping = {}
        self.mapping_flows = {}
        self.mapping_loads = {}

    def add_mapping(self, req, mapping, flow, load):
        if mapping.name in self.mapping_flows:
            raise ValueError("Received mapping with duplicate name {}".format(mapping.name))
        self.request_mapping.setdefault(req, []).append(mapping)
        self.mapping_flows[mapping.name] = flow
        self.mapping_loads[mapping.name] = load

    def validate_solution(self):
        """ validates types, latency and capacity for each request and given
        list of mappings in scenario
        """
        if (not self.scenario.validate_types()):
            return False
        for request in self.scenario.requests:
            if (request in self.request_mapping):
                for mapping in self.request_mapping[request]:
                    substrate = self.scenario.substrate
                    rules = [self.type_check(request, mapping, substrate),
                             self.latency_check(request, mapping, substrate),
                             self.capacity_check(request, mapping, substrate)]
                if not all(rules):
                    return False
        return True

    def type_check(self, request, mapping, substrate):
        """ checks if requested types are in supported types from substrate
        """
        for i in mapping.mapping_nodes:
            if (request.node[i]['type'] not in
                    substrate.node[mapping.mapping_nodes[i]]['supported_types']):
                print "Node:", mapping.mapping_nodes[i], " does not support type:", request.node[i]['type']
                return False
        return True

    def latency_check(self, request, mapping, substrate):
        for path in mapping.request.graph['latency_requirement'].keys():
            sum = 0
            for ve in path:
                sum += substrate.get_path_latency(mapping.mapping_edges[ve])
                if sum > request.graph['latency_requirement'][path]:
                    print "Latency Check failed", sum
                    return False
        return True

    def capacity_check(self, request, mapping, substrate):
        """ checks if demand of all request nodes and edges is fullfilled by
        substrate capacity
        """
        for i in mapping.mapping_nodes:
            if request.node[i]['demand'] > substrate.node[mapping.mapping_nodes[i]]['capacity']:
                s = "demand of Node {} is {} - higher than capacity of mapped node {} with capacity {}".format(
                    i, request.node[i]['demand'],
                    mapping.mapping_nodes[i],
                    mapping.mapping_nodes[i]['capacity']
                )
                print s
                return False
        for ve in mapping.request.edge:
            if mapping.mapping_edges[ve]:
                if request.edge[ve]['demand'] > substrate.get_path_capacity(mapping.mapping_edges[ve]):
                    print mapping.mapping_edges[ve]
                    s = "demand of edge {} is {} - higher than capacity of mapped edge {} with capacity {}".format(ve, request.edge[ve]['demand'],
                                                                                                                   mapping.mapping_edges[ve],
                                                                                                                   substrate.get_path_capacity(mapping.mapping_edges[ve]))
                    print s
                    return False
        return True

    def __str__(self):
        s = ""
        for req in self.request_mapping:
            s += "\n\t" + str(req)
            for mapping in self.request_mapping[req]:
                s += "\n\t" + str(mapping)
        return "FractionalScenarioSolution {} for:{}".format(self.name, s)


class Mapping(object):
    def __init__(self, name, request, substrate, is_embedded):
        self.name = name
        self.request = request
        self.substrate = substrate
        self.mapping_nodes = {}
        self.mapping_edges = {}
        self.is_embedded = is_embedded

    def lookup_request(self, u):
        """ given a substrate node u - return the mapped request node else None
        """
        try:
            return (key for key, value in self.mapping_nodes.items() if value == u).next()
        except:
            raise Exception("No mapping found for substrate node {}".format(u))

    def map_node(self, i, u):
        # check if i is a request node and u is a substrate node
        if i in self.mapping_nodes:
            raise MappingError("Tried adding duplicate node mapping of {} onto {}".format(i, u))
        if (i in self.request.nodes and u in self.substrate.nodes):
            # check wether type is supported by substrate node and if it's
            # allowed by request node
            if (self.request.node[i]['type'] in
                    self.substrate.node[u]['supported_types']):
                if (self.request.node[i]['allowed_nodes'] is None or u in
                    self.request.node[i]['allowed_nodes']):
                    self.mapping_nodes[i] = u
                else:
                    raise MappingError("Node {} of request {} can not be mapped on substrate node {} because it is restricted to {}".format(i, self.request.name, u, self.request.node[i]['allowed_nodes']))
            else:
                raise MappingError("Request node {} needs type {} but substrate node {} does not support this type".format(i,
                                                                                                                           self.request.node[i]['type'], u))
        else:
            raise MappingError("Request Noded {} can not be mapped on substrate node {}".format(i, u))

    def map_edge(self, ve, ses):
        """ maps a virtual edge ve of a request to a path(multiple edges) se's of a
        substrate"""
        if (ve in self.request.edges and set(ses) <= self.substrate.edges):
            # empty path direct mapping
            if not ses:
                self.mapping_edges[ve] = ses
            else:
                vtail, vhead = ve
                subfirsttail, subfirsthead = ses[0]
                sublasttail, sublasthead = ses[-1]
                # check that tail, head of ve are correctly mapped on tail, head of path
                if self.mapping_nodes[vtail] == subfirsttail and self.mapping_nodes[vhead] == sublasthead:
                    # it's only single edge mapped on single edge
                    if not len(ses) > 1:
                        self.mapping_edges[ve] = ses
                    else:
                        # check wether path is a real edge path and connected
                        for i, currentedge in enumerate(ses):
                            if i < len(ses) - 1:
                                currenttail, currenthead = currentedge
                                nexttail, nexthead = ses[i + 1]
                                if not currenthead == nexttail:
                                    raise MappingError("Path {} is not connected in substrate".format(ses))
                        self.mapping_edges[ve] = ses

    def get_mapping_of_node(self, i):
        return self.mapping_nodes[i]

    def __str__(self):
        return "Mapping \"{}\" with following mappings: \n\t\tNodes {} \n\t\tEdges {} ".format(self.name, self.mapping_nodes, self.mapping_edges)


class ScenarioSolutionStorage(object):
    def __init__(self, scenario_parameter_container, execution_parameter_container):
        self.algorithm_scenario_solution_dictionary = {}  # for evaluation, it was more useful to index solutions by algorithm first
        self.algorithms = set()
        self.scenario_parameter_container = scenario_parameter_container
        self.execution_parameter_container = execution_parameter_container

    def add_solution(self, algorithm_id, scenario_id, execution_id, solution):
        log.info("Adding solution for algorithm {}, scenario {}, execution {} to storage:".format(
            algorithm_id, scenario_id, execution_id)
        )
        log.info("    {}".format(solution))

        self.algorithm_scenario_solution_dictionary.setdefault(algorithm_id, {})
        self.algorithm_scenario_solution_dictionary[algorithm_id].setdefault(scenario_id, {})

        if execution_id in self.algorithm_scenario_solution_dictionary[algorithm_id][scenario_id]:
            if self.algorithm_scenario_solution_dictionary[algorithm_id][scenario_id][execution_id] is not None:
                raise SolutionStorageError("Duplicate solution: alg.: {}, scenario: {}, execution: {} exists!".format(
                    algorithm_id, scenario_id, execution_id
                ))
        self.algorithm_scenario_solution_dictionary[algorithm_id][scenario_id][execution_id] = solution
        self.algorithms.add(algorithm_id)

    def retrieve_scenario_parameters_for_index(self, scenario_index):
        return self.scenario_parameter_container.scenario_parameter_combination_list[scenario_index]

    def get_solutions_by_algorithm(self, algorithm_id):
        return self.algorithm_scenario_solution_dictionary[algorithm_id]

    def get_solutions_by_scenario_index(self, index):
        result = {}
        for alg_id, parameter_solution_dict in self.algorithm_scenario_solution_dictionary.iteritems():
            if index in parameter_solution_dict:
                result[alg_id] = parameter_solution_dict[index]
        return result
