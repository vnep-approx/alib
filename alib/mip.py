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

import gurobipy
from gurobipy import GRB, LinExpr

from . import modelcreator, solutions


class ClassicMCFError(Exception): pass


class ClassicMCFResult(modelcreator.AlgorithmResult):
    ''' Represents the result of a Multi-Commodity Flow IP Formulation.

    Accordingly, it extends the regular AlgorithmResult by storing Gurobi status information as well as
    a temporal log, detailing the solution process of Gurobi.

    '''
    def __init__(self, solution, temporal_log, status):
        super(ClassicMCFResult, self).__init__()
        self.solution = solution
        self.temporal_log = temporal_log
        self.status = status

    def get_solution(self):
        return self.solution

    def _cleanup_references_raw(self, original_scenario):
        own_scenario = self.solution.scenario
        self.solution.scenario = original_scenario
        for i, own_req in enumerate(own_scenario.requests):
            mapping = self.solution.request_mapping[own_req]
            del self.solution.request_mapping[own_req]
            original_request = original_scenario.requests[i]
            mapping.request = original_request
            mapping.substrate = original_scenario.substrate
            self.solution.request_mapping[original_request] = mapping


class ClassicMCFModel(modelcreator.AbstractEmbeddingModelCreator):

    ''' Gurobi model to construct and solve the multi-commodity flow formulation for the VNEP.

        Important: inheriting from the AbstractEmbeddingModelCreator, only the core functionality is enabled in this class.
    '''

    ALGORITHM_ID = "ClassicMCF"

    def __init__(self, scenario, gurobi_settings=None, logger=None, optimization_callback=modelcreator.gurobi_callback):
        super(ClassicMCFModel, self).__init__(scenario=scenario, gurobi_settings=gurobi_settings, logger=logger, optimization_callback=optimization_callback)

        self.var_y = {}
        self.var_z = {}

        self.time_lp = None

    def create_variables_other_than_embedding_decision_and_request_load(self):
        # node mapping variable
        for req in self.requests:
            self.var_y[req] = {}
            for vnode in req.nodes:
                self.var_y[req][vnode] = {}
                for snode in self.substrate.nodes:
                    supported_type = req.get_type(vnode) in self.substrate.get_supported_node_types(snode)
                    allowed_nodes = req.get_allowed_nodes(vnode)
                    is_allowed = allowed_nodes is None or snode in allowed_nodes
                    if supported_type and is_allowed:
                        variable_name = modelcreator.construct_name("y",
                                                                    req_name=req.name,
                                                                    vnode=vnode,
                                                                    snode=snode)

                        self.var_y[req][vnode][snode] = self.model.addVar(lb=0.0,
                                                                          ub=1.0,
                                                                          obj=0.0,
                                                                          vtype=GRB.BINARY,
                                                                          name=variable_name)

        # flow variable
        for req in self.requests:
            self.var_z[req] = {}
            for (i, j) in req.edges:
                self.var_z[req][(i, j)] = {}
                for (u, v) in self.substrate.edges:
                    variable_name = modelcreator.construct_name("z",
                                                                req_name=req.name,
                                                                vedge=(i, j),
                                                                sedge=(u, v))
                    self.var_z[req][(i, j)][(u, v)] = self.model.addVar(lb=0.0,
                                                                        ub=1.0,
                                                                        obj=0.0,
                                                                        vtype=GRB.BINARY,
                                                                        name=variable_name)

        self.model.update()

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        self.create_constraints_node_mapping()
        self.create_constraints_forbidden_edges()

        self.create_constraints_flow_preservation_and_induction()

        self.create_constraints_compute_node_load()

        self.create_constraints_compute_edge_load()

    def create_constraints_node_mapping(self):
        # node mapping
        for req in self.requests:
            for i in req.nodes:
                expr = LinExpr(
                    [(-1.0, self.var_embedding_decision[req])] +
                    [(1.0, self.var_y[req][i][snode]) for snode
                     in self.var_y[req][i].keys()]
                )
                constr_name = modelcreator.construct_name("flow_induction",
                                                          req_name=req.name,
                                                          vnode=i)  # Matthias: changed to conform to standard naming
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def create_constraints_forbidden_edges(self):
        for req in self.requests:
            for ij in req.edges:
                allowed_edges = req.edge[ij].get("allowed_edges")
                if allowed_edges is None:
                    continue
                allowed = set(allowed_edges)
                forbidden = [uv for uv in self.substrate.edges if uv not in allowed]
                for uv in forbidden:
                    constr_name = modelcreator.construct_name("forbid_edge_mapping",
                                                              req_name=req.name, vedge=ij, sedge=uv)
                    expr = LinExpr([(1.0, self.var_z[req][ij][uv])])
                    self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def create_constraints_flow_preservation_and_induction(self):
        for req in self.requests:
            for (i, j) in req.edges:
                for u in self.substrate.nodes:
                    right_expr = LinExpr()
                    if u in self.var_y[req][i]:
                        right_expr.addTerms(1.0, self.var_y[req][i][u])
                    if u in self.var_y[req][j]:
                        right_expr.addTerms(-1.0, self.var_y[req][j][u])

                    ij_mapping_vars = self.var_z[req][(i, j)]
                    left_outgoing = LinExpr([(1.0, ij_mapping_vars[sedge]) for sedge in self.substrate.out_edges[u]])
                    left_incoming = LinExpr([(1.0, ij_mapping_vars[sedge]) for sedge in self.substrate.in_edges[u]])
                    left_expr = LinExpr(left_outgoing - left_incoming)
                    constr_name = modelcreator.construct_name("flow_pres", req_name=req.name, vedge=(i, j), snode=u)  # Matthias: changed to conform to standard naming
                    self.model.addConstr(left_expr, GRB.EQUAL, right_expr, name=constr_name)

    def create_constraints_compute_node_load(self):
        # track node loads
        for req in self.requests:
            for (t, u) in self.substrate.substrate_node_resources:
                expr_terms = []
                for i in req.nodes:
                    i_type = req.get_type(i)
                    i_demand = req.get_node_demand(i)
                    if i_type == t and u in self.var_y[req][i]:
                        expr_terms.append(LinExpr(i_demand, self.var_y[req][i][u]))

                expr_terms.append(LinExpr(-1.0, self.var_request_load[req][(t, u)]))
                constr_name = modelcreator.construct_name("compute_request_node_load", req_name=req.name,
                                                          snode=u, other=t)

                expr = gurobipy.quicksum(expr_terms)
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def create_constraints_compute_edge_load(self):
        # track edge loads
        for req in self.requests:
            for sedge in self.substrate.substrate_edge_resources:
                expr_terms = []
                for vedge in req.edges:
                    vedge_demand = req.get_edge_demand(vedge)
                    expr_terms.append(LinExpr(vedge_demand, self.var_z[req][vedge][sedge]))

                constr_name = modelcreator.construct_name("compute_request_edge_load", req_name=req.name,
                                                          sedge=sedge)

                expr_terms.append(LinExpr(-1.0, self.var_request_load[req][sedge]))
                expr = gurobipy.quicksum(expr_terms)
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def recover_integral_solution_from_variables(self):
        solution_name = modelcreator.construct_name("solution_", sub_name=self.substrate.name)

        solution = solutions.IntegralScenarioSolution(solution_name, self.scenario)

        for req in self.requests:
            mapping = self.obtain_mapping(req)
            solution.add_mapping(req, mapping)

        return solution

    def post_process_integral_computation(self):
        return ClassicMCFResult(solution=self.solution, temporal_log=self.temporal_log, status=self.status)

    def obtain_mapping(self, req):
        is_embedded = self.var_embedding_decision[req].x > 0.5
        mapping_name = modelcreator.construct_name("mapping_", req_name=req.name,
                                                   sub_name=self.substrate.name)

        mapping = solutions.Mapping(mapping_name, req, self.substrate, is_embedded=is_embedded)
        if is_embedded:
            # node mapping
            for vnode in req.nodes:
                for snode, decision_var in self.var_y[req][vnode].iteritems():
                    if decision_var.x > 0.5:
                        mapping.map_node(vnode, snode)
            # edge mapping
            for (i, j) in req.edges:
                start_snode = mapping.get_mapping_of_node(i)
                end_snode = mapping.get_mapping_of_node(j)
                if start_snode == end_snode:
                    mapping.map_edge((i, j), [])
                    continue

                stack = [start_snode]
                predecessor = {}
                for enode in self.substrate.nodes:
                    predecessor[enode] = None

                while len(stack) > 0:
                    current_enode = stack.pop()
                    if current_enode == end_snode:
                        del stack
                        break
                    for out_edge in self.substrate.out_edges[current_enode]:
                        tail, head = out_edge
                        if self.var_z[req][(i, j)][out_edge].X > 0.5 and predecessor[head] is None:
                            stack.append(head)
                            predecessor[head] = tail
                if predecessor[end_snode] is None:
                    raise Exception("Never possible")  # TODO..
                # reconstruct path
                path = []
                current_enode = end_snode
                while current_enode != start_snode:
                    # return mapping
                    previous_hop = predecessor[current_enode]
                    path.append((previous_hop, current_enode))
                    current_enode = previous_hop
                # reverse edges such that path leads from super source to super sink
                path.reverse()
                mapping.map_edge((i, j), path)
        return mapping

    def fix_mapping_variables_according_to_integral_solution(self, solution):
        if not isinstance(solution, solutions.IntegralScenarioSolution):
            msg = "Expected solutions.IntegralScenarioSolution instance, received {} of type {}".format(solution, type(solution))
            raise TypeError(msg)
        if solution.scenario is not self.scenario:
            msg = "This method requires that the solution is based on the same scenario as the Modelcreator."
            raise ClassicMCFError(msg)
        for req in self.requests:
            mapping = solution.request_mapping[req]
            self._fix_embedding_variable(req, mapping)
            if not mapping.is_embedded:
                continue
            for i in req.nodes:
                for u in req.get_allowed_nodes(i):
                    fix_i_u_mapping_constraint = LinExpr([(1.0, self.var_y[req][i][u])])
                    name = "{req}_fix_{i}_{u}".format(req=req.name, i=i, u=u)
                    if u == mapping.get_mapping_of_node(i):
                        self.model.addConstr(fix_i_u_mapping_constraint, GRB.EQUAL, 1.0, name=name)
                    else:
                        self.model.addConstr(fix_i_u_mapping_constraint, GRB.EQUAL, 0.0, name=name)
            for ij in req.edges:
                i, j = ij
                for uv in self.substrate.edges:
                    u, v = uv
                    fix_ij_uv_mapping_constraint = LinExpr([(1.0, self.var_z[req][ij][uv])])
                    name = "{}_fix_{}_{}__{}_{}".format(req.name, i, j, u, v)
                    if uv in mapping.mapping_edges[ij]:
                        self.model.addConstr(fix_ij_uv_mapping_constraint, GRB.EQUAL, 1.0, name=name)
                    else:
                        self.model.addConstr(fix_ij_uv_mapping_constraint, GRB.EQUAL, 0.0, name=name)

    def _fix_embedding_variable(self, req, mapping):
        force_embedding_constraint = LinExpr([(1.0, self.var_embedding_decision[req])])
        name = modelcreator.construct_name("force_embedding", req_name=req.name)
        if mapping.is_embedded:
            self.model.addConstr(force_embedding_constraint, GRB.EQUAL, 1.0, name=name)
        else:
            self.model.addConstr(force_embedding_constraint, GRB.EQUAL, 0.0, name=name)

