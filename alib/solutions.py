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

from . import util

log = util.get_logger(__name__, make_file=False, propagate=True)


class SolutionStorageError(Exception):    pass


class MappingError(Exception): pass


class IntegralScenarioSolution(object):
    ''' Represents an integral solution to a scenario, i.e. it indicates for a subset (or all) requests integral mappings.

    '''

    def __init__(self, name, scenario):
        self.name = name
        self.scenario = scenario
        # this a dictionary of request --> mapping
        self.request_mapping = {}

    def add_mapping(self, req, mapping):
        self.request_mapping[req] = mapping

    def validate_solution(self):
        """ validates types and capacity for each request and given mapping in scenario
        """
        if (not self.scenario.validate_types()):
            return False
        for request in self.scenario.requests:
            if (request in self.request_mapping and self.request_mapping[request] is not None):
                mapping = self.request_mapping[request]
                substrate = self.scenario.substrate
                rules = [self.type_check(request, mapping, substrate),
                         self.capacity_check(request, mapping, substrate)]
                if not all(rules):
                    return False
        return True

    def validate_solution_fulfills_capacity(self):
        result = True
        substrate = self.scenario.substrate
        substrate_resources = {uv: substrate.edge[uv]["capacity"] for uv in substrate.edges}
        for u in substrate.nodes:
            for ntype in substrate.node[u]["supported_types"]:
                substrate_resources[ntype, u] = substrate.node[u]["capacity"][ntype]
        for req, mapping in self.request_mapping.items():
            if mapping is None:
                continue
            for i, u in mapping.mapping_nodes.items():
                t = req.get_type(i)
                demand = req.get_node_demand(i)
                substrate_resources[t, u] -= demand
            for ij, uv_list in mapping.mapping_edges.items():
                demand = req.get_edge_demand(ij)
                for uv in uv_list:
                    substrate_resources[uv] -= demand
        for res, remaining_cap in substrate_resources.items():
            if remaining_cap < 0:
                log.error("resource {} violated capacity by {}".format(res, -remaining_cap))
                result = False
        return result

    def type_check(self, request, mapping, substrate):
        """ checks if requested types are in supported types from substrate
        """
        for i in mapping.mapping_nodes:
            i_type = request.get_type(i)
            u = mapping.mapping_nodes[i]
            if i_type not in substrate.get_supported_node_types(u):
                print "Node {} does not support type {}".format(u, request.node[i]['type'])
                return False
        return True

    def capacity_check(self, request, mapping, substrate):
        """ checks if demand of all request nodes and edges is fullfilled by
        substrate capacity
        """
        for i, u_i in mapping.mapping_nodes.items():
            i_demand = request.get_node_demand(i)
            i_type = request.get_type(i)
            u_i_capacity = substrate.get_node_type_capacity(u_i, i_type)
            if u_i_capacity < i_demand:
                return False
        for ij in mapping.request.edge:
            i, j = ij
            mapped_path = mapping.mapping_edges[ij]
            if not mapped_path and mapping.mapping_nodes[i] == mapping.mapping_nodes[j]:  # end nodes are mapped to same node
                continue
            ij_demand = request.get_edge_demand(ij)
            if substrate.get_path_capacity(mapped_path) < ij_demand:
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
    ''' Scenario solution in which for each request convex combinations of mappings are allowed.

    '''

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
        """ validates types and capacity for each request and given
        list of mappings in scenario
        """
        if (not self.scenario.validate_types()):
            return False
        for request in self.scenario.requests:
            if (request in self.request_mapping):
                for mapping in self.request_mapping[request]:
                    substrate = self.scenario.substrate
                    rules = [self.type_check(request, mapping, substrate),
                             self.capacity_check(request, mapping, substrate)]
                if not all(rules):
                    return False
        return True

    def type_check(self, request, mapping, substrate):
        """ checks if requested types are in supported types from substrate
        """
        for i in mapping.mapping_nodes:
            if (request.get_type(i) not in substrate.node[mapping.mapping_nodes[i]]['supported_types']):
                print "Node:", mapping.mapping_nodes[i], " does not support type:", request.node[i]['type']
                return False
        return True

    def capacity_check(self, request, mapping, substrate):
        """ checks if demand of all request nodes and edges is fullfilled by
        substrate capacity
        """
        for i in mapping.mapping_nodes:
            i_type = request.get_type(i)
            i_demand = request.get_node_demand(i)
            u_i = mapping.mapping_nodes[i]
            u_i_capacity = substrate.get_node_type_capacity(u_i, i_type)
            if u_i_capacity < i_demand:
                s = "demand of Node {} is {} - higher than capacity of mapped node {} with capacity {}".format(
                    i, i_demand, u_i, u_i_capacity
                )
                print s
                return False
        for ij in mapping.request.edges:
            mapped_path = mapping.mapping_edges[ij]
            if mapped_path:
                path_capacity = substrate.get_path_capacity(mapped_path)
                ij_demand = request.get_edge_demand(ij)
                if path_capacity < ij_demand:
                    print mapped_path
                    print "Demand of edge {} is {} - higher than capacity of mapped path {} with capacity {}".format(
                        ij, ij_demand, mapped_path, path_capacity
                    )
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
    ''' Represents a valid mapping of a single request on a sustrate.

        Initially, the mapping is empty and needs to be populated by the map_node and map_edge functions.
    '''

    def __init__(self, name, request, substrate, is_embedded):
        self.name = name
        self.request = request
        self.substrate = substrate
        self.mapping_nodes = {}
        self.mapping_edges = {}
        self.is_embedded = is_embedded

    def map_node(self, i, u):
        ''' Maps the single request node i on to the substrate node u.
        '''
        if i in self.mapping_nodes:
            raise MappingError("Tried adding duplicate node mapping of {} onto {}. (already mapped to {})".format(i, u, self.mapping_nodes[i]))
        if i not in self.request.nodes:
            raise MappingError("Request node {} does not exist! (Tried mapping on substrate node {})".format(i, u))
        if u not in self.substrate.nodes:
            raise MappingError("Substrate node {} does not exist! (Tried mapping request node {})".format(u, i))

        i_type = self.request.get_type(i)
        u_types = self.substrate.get_supported_node_types(u)
        if i_type not in u_types:
            raise MappingError("Request node {} needs type {} but substrate node {} does not support this type".format(
                i, self.request.node[i]['type'], u
            ))
        i_allowed_nodes = self.request.get_allowed_nodes(i)
        if i_allowed_nodes is None or u in i_allowed_nodes:
            self.mapping_nodes[i] = u
        else:
            raise MappingError("Node {} of request {} cannot be mapped on substrate node {} because it is restricted to {}".format(
                i, self.request.name, u, i_allowed_nodes
            ))

    def map_edge(self, ij, mapped_path):
        """ maps a virtual edge ij of the request to a path(multiple edges) mapped_path of the substrate"""

        if ij in self.mapping_edges:
            raise MappingError("Tried adding duplicate edge mapping of {} onto {}. (already mapped to {})".format(ij, mapped_path, self.mapping_edges[ij]))
        if ij not in self.request.edges:
            raise MappingError("Request edge {} does not exist!".format(ij))
        if not set(mapped_path) <= self.substrate.edges:
            raise MappingError("Mapping for {} contains edges not in the substrate!".format(ij, set(mapped_path) - self.substrate.edges))

        # empty path direct mapping
        if not mapped_path:
            self.mapping_edges[ij] = mapped_path
        else:
            i, j = ij
            subfirsttail, subfirsthead = mapped_path[0]
            sublasttail, sublasthead = mapped_path[-1]
            # check that tail, head of ve are correctly mapped on tail, head of path
            if self.mapping_nodes[i] == subfirsttail and self.mapping_nodes[j] == sublasthead:
                # it's only single edge mapped on single edge
                if not len(mapped_path) > 1:
                    self.mapping_edges[ij] = mapped_path
                else:
                    # check wether path is a real edge path and connected
                    for i, currentedge in enumerate(mapped_path):
                        if i < len(mapped_path) - 1:
                            currenttail, currenthead = currentedge
                            nexttail, nexthead = mapped_path[i + 1]
                            if currenthead != nexttail:
                                raise MappingError("Path {} is not connected in substrate".format(mapped_path))
                    self.mapping_edges[ij] = mapped_path

    def get_mapping_of_node(self, i):
        return self.mapping_nodes[i]

    def __str__(self):
        return "Mapping \"{}\" with following mappings: \n\t\tNodes {} \n\t\tEdges {} ".format(self.name, self.mapping_nodes, self.mapping_edges)


class ScenarioSolutionStorage(object):
    ''' Encapsulates the solutions for a whole set of scenarios.

        In general, this storage mirrors the ScenarioParametersContainer: for each scenario of the container, one
        solution (should be) is contained in the solution storage.
    '''

    def __init__(self, scenario_parameter_container, execution_parameter_container):
        # stores solutions for each algorithm and each scenario; dict --> dict --> dict
        self.algorithm_scenario_solution_dictionary = {}
        # stores the algorithm identifiers for which solutions are contained
        self.algorithms = set()
        # the container with the original scenarios and its generation parameters
        self.scenario_parameter_container = scenario_parameter_container
        # the container with the execution specifications
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

    def merge_with_other_sss(self, other_sss):
        if not isinstance(other_sss, ScenarioSolutionStorage):
            raise ValueError("Can only merge with other ScenarioSolutionStorage, received {}.".format(other_sss))
        if self.execution_parameter_container.execution_parameter_space != other_sss.execution_parameter_container.execution_parameter_space:
            raise ValueError("Other ScenarioSolutionStorage has incompatible execution parameters.")

        self.scenario_parameter_container.merge_with_other_scenario_parameter_container(
            other_sss.scenario_parameter_container
        )

        alg_sol_dict = other_sss.algorithm_scenario_solution_dictionary
        for algorithm_id, scenario_execution_solution_dict in alg_sol_dict.iteritems():
            for scenario_id, execution_solution_dict in scenario_execution_solution_dict.iteritems():
                for execution_id, solution in execution_solution_dict.iteritems():
                    self.add_solution(
                        algorithm_id=algorithm_id,
                        scenario_id=scenario_id,
                        execution_id=execution_id,
                        solution=solution,
                    )
