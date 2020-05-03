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

import pkg_resources

import cPickle as pickle
import copy
import networkx as nx
import itertools
import glob
import math
import multiprocessing as mp
from multiprocessing.managers import BaseManager, SyncManager
import os
from unidecode import unidecode

import time
import yaml
from collections import deque, namedtuple
from random import Random


import numpy.random

from . import datamodel, mip, modelcreator, util

global_logger = util.get_logger(__name__, make_file=False, propagate=True)

random = Random("scenariogeneration")

numpy.random.seed(1234)

UNIVERSAL_NODE_TYPE = "universal"

DATA_PATH = pkg_resources.resource_filename("alib", "data/")


class RequestGenerationError(Exception): pass


class SubstrateReaderError(Exception): pass


class ScenarioGeneratorError(Exception): pass


class ScenarioParameterError(Exception): pass


class ExperimentSpecificationError(Exception): pass


#  These are the required tasks that must be adressed when generating scenarios
SUBSTRATE_GENERATION_TASK = "substrate_generation"
REQUEST_GENERATION_TASK = "request_generation"
PROFIT_CALCULATION_TASK = "profit_calculation"
NODE_PLACEMENT_TASK = "node_placement_restriction_mapping"
SCENARIO_GENERATION_TASKS = [
    SUBSTRATE_GENERATION_TASK,
    REQUEST_GENERATION_TASK,
    PROFIT_CALCULATION_TASK,
    NODE_PLACEMENT_TASK
]
REQUIRED_TASKS = {
    SUBSTRATE_GENERATION_TASK,
    REQUEST_GENERATION_TASK
}


def verify_completeness_of_scenario_parameters(scenario_parameter_space):
    ''' Checks scenario parameters for completeness and raises a warnings and exceptions if necessary.

    :param scenario_parameter_space:
    :return: nothing
    '''
    errors = []
    warnings = []
    for task in SCENARIO_GENERATION_TASKS:
        if task not in scenario_parameter_space:
            if task in REQUIRED_TASKS:
                errors.append("Scenario parameters do not address the required scenario generation task {}".format(task))
            else:
                warnings.append("Scenario parameters do not address the optional scenario generation task {}".format(task))
    for task in scenario_parameter_space:
        if len(scenario_parameter_space[task]) == 0:
            errors.append("Scenario parameters require task {}, but do not provide a strategy for it!".format(task))
            continue
        for strategy_dict in scenario_parameter_space[task]:
            strategy = strategy_dict.keys()[0]
            class_param_dict = strategy_dict[strategy]
            strategy_class_name = class_param_dict.keys()[0]
            if strategy_class_name not in globals():
                errors.append("Could not resolve class {}, employed in strategy {} for task {}.".format(
                    strategy_class_name, strategy, task
                ))
                continue
            strategy_class = globals()[strategy_class_name]
            if not hasattr(strategy_class, "EXPECTED_PARAMETERS"):
                warnings.append("Class {strategy_class}, employed in strategy {strategy} for task {task}, does not specify which parameters it requires.".format(
                    strategy_class=strategy_class.__name__,
                    strategy=strategy,
                    task=task
                ))
                continue
            expected = set(strategy_class.EXPECTED_PARAMETERS)
            parameters = set(class_param_dict.values()[0].keys())
            if expected - parameters:
                msg = "The following parameters for {task}, {strategy} were not defined but are required by {strategy_class}:\n        {missing}".format(
                    task=task,
                    strategy=strategy,
                    strategy_class=strategy_class,
                    missing=", ".join(expected - parameters)
                )
                errors.append(msg)
            if parameters - expected:
                msg = "The following parameters for {task}, {strategy} were defined but are not required by {strategy_class}:\n        {missing}".format(
                    task=task,
                    strategy=strategy,
                    strategy_class=strategy_class,
                    missing=", ".join(parameters - expected)
                )
                warnings.append(msg)
    if warnings:
        global_logger.warning("Warning(s): \n  - " + "\n  - ".join(warnings))
    if errors:
        raise ExperimentSpecificationError("Error(s):\n  - " + "\n  - ".join(errors))


def generate_pickle_from_yml(parameter_file, scenario_out_pickle, threads=1, scenario_index_offset=0):
    ''' main function to generate a scenario pickle from a parameter file

    :param parameter_file: yaml file detailing the scenario parameterds
    :param scenario_out_pickle: output file to write the pickle to
    :param threads: number of threads that shall be used for generating the scenarios
    :param scenario_index_offset: offset of scenario indicices to enable merging of distinct scenario storages
    :return: None
    '''
    param_space = yaml.load(parameter_file)
    sg = ScenarioGenerator(threads)
    repetition = 1
    if 'scenario_repetition' in param_space:
        repetition = param_space['scenario_repetition']
        del param_space['scenario_repetition']
    sg.generate_scenarios(param_space, repetition, scenario_index_offset=scenario_index_offset)
    container = sg.scenario_parameter_container
    out = os.path.abspath(os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                       scenario_out_pickle))
    with open(out, "wb") as f:
        pickle.dump(container, f)


class ScenarioParameterContainer(object):
    '''Represents a set of scenarios accessible via its parameters according to which the scenarios (instances) were generated.

    '''

    def __init__(self, scenario_parameter_room, scenario_index_offset=0):
        self.scenarioparameter_room = scenario_parameter_room
        self.scenario_index_offset = scenario_index_offset
        self.scenario_list = []
        self.scenario_parameter_combination_list = []
        self.scenario_parameter_dict = {}
        self.scenario_triple = {}

    def generate_all_scenario_parameter_combinations(self, repetition=1):
        """
        Given a dictionary representing the parameter space of an experiment,
        this function generates a list of parameter dictionaries, each of which
        can be used by the ScenarioGenerator to generate a single scenario.

        :param repetition: how many times each scenario will be builded (Default value = 1)
        :return: A list of dictionaries, where each dictionary specifies a single scenario
            {generation_task -> {strategy_name -> {class -> {parameter -> [values]}}}}
        """
        product_dict = {}
        # Expand the inner-most parameters  (dict of lists -> list of dicts)
        for scenario_generation_task, scenario_task_strategy_list in self.scenarioparameter_room.iteritems():
            product_dict[scenario_generation_task] = {}
            for scenario_task_strategy in scenario_task_strategy_list:
                # TODO: make this more clear, and add some sanity checks ensuring that scenario_task_strategy contains exactly 1 strategy
                strategy_name = scenario_task_strategy.keys()[0]
                strategy_class = scenario_task_strategy.values()[0].keys()[0]
                strategy_parameter_space = scenario_task_strategy.values()[0].values()[0]
                self.make_values_immutable(strategy_parameter_space)
                inner_parameter_combinations = ScenarioParameterContainer._expand_innermost_parameter_space(strategy_parameter_space)
                expanded_solution_parameters = {strategy_class: inner_parameter_combinations}
                product_dict[scenario_generation_task][strategy_name] = expanded_solution_parameters
        # A list of all tasks that need to be completed during scenario generation, e.g.:
        # [t1, t2, t3]
        tasks = sorted(product_dict.keys())
        result = []
        # tuple of lists of the user-defined strategy names for each task, e.g.:
        # ([t1_s1, t1_s2], [t2_s1], [t3_s1, t3_s2, t3_s3])
        tuple_of_solutions_for_each_task = tuple(product_dict[t].keys() for t in tasks)

        # list of tuples of strategy names, containing exactly one strategy for each task, e.g.:
        # [(t1_s1, t2_s1, t3_s1), (t1_s2, t2_s1, t3_s1),
        #  (t1_s1, t2_s1, t3_s2), (t1_s2, t2_s1, t3_s2),
        #  (t1_s1, t2_s1, t3_s3), (t1_s2, t2_s1, t3_s3)]
        valid_strategy_combination_list = itertools.product(*tuple_of_solutions_for_each_task)

        for strategy_combination in valid_strategy_combination_list:
            # class_list: for each strategy in the strategy_combination, the class_list contains the
            # name of the class where the strategy is implemented, e.g.:
            # strategy_combination = ("service_chains", "optimal_profit_calc", ...)
            # => class_list = ["ServiceChainGenerator", "OptimalEmbeddingProfitCalculator", ...]
            class_list = [product_dict[tasks[i]][sol].keys()[0] for (i, sol) in enumerate(strategy_combination)]

            # tuple_task_parameter_list contains the parameter space for each
            tuple_task_parameter_list = tuple(product_dict[tasks[i]][sol].values()[0] for (i, sol) in enumerate(strategy_combination))

            # expand the parameter spaces of all strategies to parameter dictionaries:
            for combination in itertools.product(*tuple_task_parameter_list):
                single_parameter_dict = dict()
                for (i, task) in enumerate(tasks):
                    single_parameter_dict[task] = {}
                    single_parameter_dict[task][strategy_combination[i]] = {class_list[i]: combination[i]}
                for rep in range(0, repetition):
                    copydict = dict(single_parameter_dict)
                    copydict['repetition'] = rep
                    copydict['maxrepetition'] = repetition
                    result.append(copydict)
        self.scenario_parameter_combination_list = result
        return result

    @staticmethod
    def _expand_innermost_parameter_space(parameter_space):
        """

        :param self:
        :param parameter_space: dictionar
        :return:
        """
        all_parameters = sorted(parameter_space.keys())
        parameter_combinations = [
            product for product in
            itertools.product(*(parameter_space[parameter]
                                for parameter in all_parameters))
        ]
        return [dict(combination) for combination in
                [zip(all_parameters, product) for product in parameter_combinations]]

    def make_values_immutable(self, raw_parameter):
        """
        This converts list parameters to tuples, e.g.
        branching_distribution: [[0.0, 0.5, 0.5]] becomes
        branching_distribution: [(0.0, 0.5, 0.5)]
        
        :param raw_parameter:
        :return:
        """
        for key, value_list in raw_parameter.iteritems():
            if isinstance(value_list[0], list):
                raw_parameter[key] = [tuple(x) for x in value_list]

    def merge_with_other_scenario_parameter_container(self, other):
        """
        self.scenarioparameter_room =
        self.scenario_list = []
        self.scenario_parameter_combination_list = []
        self.scenario_parameter_dict = {}
        self.scenario_triple = {}
        """
        overlap = set(self.scenario_triple.keys()).intersection(other.scenario_triple.keys())
        if overlap:
            msg = "Cannot merge scenario parameter containers due to overlapping scenario IDs {}".format(
                overlap
            )
            raise ScenarioGeneratorError(msg)
        self.scenario_list += other.scenario_list
        self.scenario_parameter_combination_list += other.scenario_parameter_combination_list
        self.scenario_triple.update(other.scenario_triple)

        for i, (sp, scenario) in other.scenario_triple.items():
            self.fill_reverselookup_dict(sp, i)

        if not isinstance(self.scenarioparameter_room, list):
            self.scenarioparameter_room = [self.scenarioparameter_room]
        if not isinstance(other.scenarioparameter_room, list):
            other.scenarioparameter_room = [other.scenarioparameter_room]
        self.scenarioparameter_room += other.scenarioparameter_room

    def init_reverselookup_dict(self):
        for task in SCENARIO_GENERATION_TASKS:
            self.scenario_parameter_dict.setdefault(task, dict())

    def fill_reverselookup_dict(self, sp, currentindex):
        spd = self.scenario_parameter_dict
        for task in SCENARIO_GENERATION_TASKS:
            if task not in sp:
                continue
            strat_id = sp[task].keys()[0]
            spd[task].setdefault(strat_id, dict())
            spd[task][strat_id].setdefault('all', set())
            spd[task][strat_id]['all'].add(currentindex)
            # spd[task][sp[task].keys()[0]] = sp[task][sp[task].keys()[0]]
            for strat in sp[task]:
                spd[task][strat].setdefault(sp[task][strat].keys()[0], dict())
                for class_name in sp[task][strat]:
                    spd[task][strat].setdefault(class_name, dict())
                    for key, val in sp[task][strat][class_name].iteritems():
                        spd[task][strat][class_name].setdefault(key, dict())
                        spd[task][strat][class_name][key].setdefault(val, set())
                        spd[task][strat][class_name][key][val].add(currentindex)


class CustomizedDataManager(SyncManager):
    pass


CustomizedDataManager.register('UndirectedGraphStorage', datamodel.UndirectedGraphStorage)


class ScenarioGenerator(object):
    '''Class to generate scenarios according to a specific parameter space.

    '''

    def __init__(self, threads=1):
        self.scenario_parameter_container = None
        self.threads = threads
        self.repetition = 1
        self.main_data_manager = None
        self.actual_data_managers = {}

    def generate_scenarios(self, scenario_parameter_space, repetition=1, scenario_index_offset=0):
        self.repetition = repetition
        global_logger.info("Generating scenarios...")

        if "data_managers" in scenario_parameter_space:
            self.main_data_manager = CustomizedDataManager()
            self.main_data_manager.start()
            for key, value in scenario_parameter_space["data_managers"].iteritems():
                if key == "UndirectedGraphStorage":
                    global_logger.info("Starting UndirectedGraphStorage Manager")
                    graph_storage_manager = self.main_data_manager.UndirectedGraphStorage(parameter_name=None)
                    graph_storage_lock = self.main_data_manager.Lock()
                    global_logger.info("\tLoading UndirectedGraphStorage data")
                    graph_storage_manager.load_from_pickle(value)
                    global_logger.info("\tDone.")
                    self.actual_data_managers[key] = (graph_storage_manager, graph_storage_lock)
                else:
                    raise ValueError("Data Manager {} is unknown.".format(key))
            del scenario_parameter_space['data_managers']

        self.scenario_parameter_container = ScenarioParameterContainer(scenario_parameter_space, scenario_index_offset=scenario_index_offset)
        self.scenario_parameter_container.generate_all_scenario_parameter_combinations(repetition)
        scenario_parameter_combination_list = self.scenario_parameter_container.scenario_parameter_combination_list
        self.scenario_parameter_container.init_reverselookup_dict()
        if self.threads > 1:
            iterator = self._multiprocessed(scenario_parameter_combination_list, scenario_index_offset=scenario_index_offset)
        else:
            iterator = self._singleprocessed(scenario_parameter_combination_list, scenario_index_offset=scenario_index_offset)
        for i, scenario, sp in iterator:
            self.scenario_parameter_container.fill_reverselookup_dict(sp, i)
            self.scenario_parameter_container.scenario_list.append(scenario)
            self.scenario_parameter_container.scenario_triple[i] = (sp, scenario)
        return self.scenario_parameter_container.scenario_triple

    def _singleprocessed(self, scenario_parameter_combination_list, scenario_index_offset=0):
        for i, sp in enumerate(scenario_parameter_combination_list, scenario_index_offset):
            yield build_scenario((i, sp, self.actual_data_managers))

    def _multiprocessed(self, scenario_parameter_combination_list, scenario_index_offset=0):
        proc_pool = mp.Pool(processes=self.threads, maxtasksperchild=100)
        task_list = [(i + scenario_index_offset, scenario_parameter_combination_list[i], self.actual_data_managers) for i in range(len(scenario_parameter_combination_list))]

        for out in proc_pool.map(build_scenario, task_list):
            yield out
        proc_pool.close()
        proc_pool.join()


def instantiate_class_from_name_space_dicts(class_name, class_kwargs_dict, names_space_dicts):
    class_instance = None
    for ns in names_space_dicts:
        if class_name in ns:
            class_instance = ns[class_name](**class_kwargs_dict)
    if class_instance is None:
        raise ScenarioGeneratorError("Class {} cannot be instantiated from any of the given namespaces: {}".
                                     format(class_name, names_space_dicts))
    return class_instance


def build_scenario(i_sp_tup):
    """
    Build a single scenario based on the scenario parameters.

    This function performs the scenario generation steps in the correct order:
    
    1. generation of the substrate topologies (including capacities) from the topology zoo
    2. generation of the request topologies (including resource demands)
    3. (optional) restrict the allowed node mappings
    4. (optional) calculate each request's profit. The scenario's objective will
       be set to profit maximization if a profit calculator is given, otherwise it
       defaults to cost minimization.

    The function also creates a separate logger to maintain readability when the scenario
    generation is done by multiple threads.

    :param i_sp_tup: Tuple containing the index of the scenario as first, and the scenario parameters
                     as returned by ScenarioParameterContainer.generate_all_scenario_parameter_combinations
                     as second element.
    :return:
    """

    try:

        i, sp, datamanager_dict = None, None, None

        if len(i_sp_tup) == 2:
            i, sp = i_sp_tup
        elif len(i_sp_tup) == 3:
            i, sp, datamanager_dict = i_sp_tup
        else:
            raise ValueError("Don't know how to handle {} many arguments.".format(len(i_sp_tup)))

        logger = util.get_logger("sg_worker_{}".format(os.getpid()), make_file=True, propagate=False)
        logger.info("Generating scenario {}  with {}".format(i, sp))
        scenario = datamodel.Scenario(name="scenario_{}_rep_{}".format(i / sp['maxrepetition'], sp['repetition']),
                                      substrate=None,
                                      requests=None,
                                      objective=datamodel.Objective.MIN_COST)
        # NOTE: not a nice way due to many reasons, but follows previous design, and keeps separation of newly implemented generators
        import scenariogeneration_for_fog_model as sg_fog_model
        all_name_spaces = [globals(), vars(sg_fog_model)]
        class_name_substrate_generator = sp[SUBSTRATE_GENERATION_TASK].values()[0].keys()[0]
        task_class_kwargs = {'logger': logger}
        subsr_gen = instantiate_class_from_name_space_dicts(class_name_substrate_generator, task_class_kwargs, all_name_spaces)
        subsr_gen.apply(sp, scenario)
        class_name_request_generator = sp[REQUEST_GENERATION_TASK].values()[0].keys()[0]
        req_gen = instantiate_class_from_name_space_dicts(class_name_request_generator, task_class_kwargs, all_name_spaces)
        if datamanager_dict is not None:
            req_gen.register_data_manager_dict(datamanager_dict)
        req_gen.apply(sp, scenario)
        if NODE_PLACEMENT_TASK in sp:
            class_name_npr = sp[NODE_PLACEMENT_TASK].values()[0].keys()[0]
            npr = instantiate_class_from_name_space_dicts(class_name_npr, task_class_kwargs, all_name_spaces)
            npr.apply(sp, scenario)
        if PROFIT_CALCULATION_TASK in sp:
            class_name_profit_calc = sp[PROFIT_CALCULATION_TASK].values()[0].keys()[0]
            pc = instantiate_class_from_name_space_dicts(class_name_profit_calc, task_class_kwargs, all_name_spaces)
            pc.apply(sp, scenario)
            scenario.objective = datamodel.Objective.MAX_PROFIT
    except Exception as e:
        import traceback
        logger.error("Error during scenario generation at worker {pid}: \n {exp}".format(pid=os.getpid(), exp=traceback.format_exc()))

    return i, scenario, sp


class ScenariogenerationTask(object):
    """
    Base class for all steps in the scenario generation process.
    Currently it just handles the instantiation of the logger.
    """

    def __init__(self, logger):
        if logger is None:
            logger = global_logger
        self.logger = logger

    def apply(self, scenario_parameters, scenario):
        """
        Apply this task to the scenario object, given the scenario_parameters.

        :param scenario_parameters:
        :param scenario:
        :return:
        """
        raise NotImplementedError("This is an abstract method! Use one of the implementations defined below.")


class AbstractRequestGenerator(ScenariogenerationTask):
    """
    Base class for the request generation task. Subclasses should implement the
    generate_request method defined below, which should return an object of the
    type datamodel.Request.
    """

    def __init__(self, logger=None):
        super(AbstractRequestGenerator, self).__init__(logger)
        self._substrate = None
        self._raw_parameters = None
        self._scenario_parameters_have_changed = True  # to avoid redundant calculations on the same scenario
        self._node_types = None
        self._data_manager_dict = None

    def register_data_manager_dict(self, data_manager_dict):
        self._data_manager_dict = data_manager_dict

    def apply(self, scenario_parameters, scenario):
        class_raw_parameters_dict = scenario_parameters[REQUEST_GENERATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise RequestGenerationError("")
        raw_parameters = class_raw_parameters_dict[class_name]
        scenario.requests = self.generate_request_list(raw_parameters,
                                                       scenario.substrate,
                                                       normalize=raw_parameters["normalize"])

    def generate_request_dictionary(self, raw_parameters, substrate, base_name="vnet_{id}", normalize=False):
        raise NotImplementedError("Not necessary at the moment")

    def generate_request_list(self, raw_parameters, substrate, base_name="vnet_{id}", normalize=False):
        requests = []
        self._scenario_parameters_have_changed = True
        self.logger.info(
            "{}: Generating request list with {} requests.".format(
                self.__class__.__name__,
                raw_parameters["number_of_requests"]
            )
        )
        for i in xrange(raw_parameters["number_of_requests"]):
            name = base_name.format(id=i + 1)
            req = self.generate_request(name, raw_parameters, substrate)
            self.logger.debug("Generated {} with {} nodes and {} edges".format(req.name, len(req.nodes), len(req.edges)))
            requests.append(req)
            self._scenario_parameters_have_changed = False  # we will generate more requests with the same parameters
        self._scenario_parameters_have_changed = True
        if normalize:
            self.normalize_resource_footprint(raw_parameters, requests, substrate)

        return requests

    def generate_request(self, name, raw_parameters, substrate):
        raise NotImplementedError("This is an abstract method! Use one of the implementations defined below.")

    def _next_node_type(self):
        return random.choice(self._node_types)

    def verify_substrate_has_sufficient_capacity(self, request, substrate):
        for i in request.nodes:
            allowed_nodes = request.node[i]["allowed_nodes"]
            demand = request.node[i]["demand"]
            ntype = request.node[i]["type"]
            max_substrate_capacity = max(substrate.get_node_type_capacity(u, ntype) for u in allowed_nodes)
            if demand > max_substrate_capacity:
                self.logger.info("Capacity limit violated by request node {} of type {} with demand {} (max. capacity = {})".format(i, ntype, demand, max_substrate_capacity))
                return False

        # TODO when we implement edge mapping restrictions, these 2 lines should move into the loop... request.get_allowed_edges(ij)
        allowed_sedges = substrate.edges
        max_substrate_capacity = max(substrate.get_edge_capacity(uv) for uv in allowed_sedges)
        for ij in request.edges:
            demand = request.edge[ij]["demand"]
            if demand > max_substrate_capacity:
                self.logger.info("Capacity limit violated by request edge {} with demand {} (max. capacity = {})".format(ij, demand, max_substrate_capacity))
                return False
        return True

    def normalize_resource_footprint(self, raw_parameters, requests, substrate):
        edge_footprint = sum(
            sum(req.get_edge_demand(ij) for ij in req.edges)
            for req in requests
        )
        node_footprint = {
            nt: sum(
                sum(req.get_node_demand(i) for i in req.get_nodes_by_type(nt))
                for req in requests)
            for nt in substrate.types
        }
        desired_edge_footprint = (1.0 / raw_parameters["edge_resource_factor"]) * substrate.get_total_edge_resources()
        desired_node_footprint = {nt: raw_parameters["node_resource_factor"] * substrate.get_total_node_resources(nt)
                                  for nt in substrate.types}
        for req in requests:
            for edge in req.edges:
                req.edge[edge]["demand"] *= (desired_edge_footprint / edge_footprint)
            for node in req.nodes:
                nt = req.node[node]["type"]
                req.node[node]["demand"] *= (desired_node_footprint[nt] / node_footprint[nt])


class ServiceChainGenerator(AbstractRequestGenerator):
    """
    Generate a Request that represents a Service Chain. The Request consists of
    a chain connecting a source "s" and target "t", with additional edges added
    at random between the intermediate nodes, according to the "probability" scenario
    parameter. If "probability" is set to 0.0, a datamodel.LinearRequest object will be
    generated instead.

    The source and target nodes are each mapped to a single substrate node.

    """

    SOURCE_NODE = "s"
    TARGET_NODE = "t"

    EXPECTED_PARAMETERS = [
        "number_of_requests",  # used for estimating average resource demand
        "min_number_of_nodes", "max_number_of_nodes",
        "probability",
        "node_resource_factor",
        "edge_resource_factor"
    ]

    def __init__(self, logger=None):
        super(ServiceChainGenerator, self).__init__(logger)
        self.average_request_node_resources_per_type = None  # dictionary mapping node types to the average resource demand of a node of the given type
        self.average_request_edge_resources = None
        self._raw_parameters = None

    def generate_request(self,
                         name,
                         raw_parameters,
                         substrate):
        self._node_types = list(substrate.types)
        self._substrate = substrate
        self._raw_parameters = raw_parameters
        self._calculate_average_resource_demands()
        req = self._generate_request_graph(name)
        # only allow request which has a solution
        while not self.verify_substrate_has_sufficient_capacity(req, substrate):
            req = self._generate_request_graph(name)

        self._scenario_parameters_have_changed = True
        self._substrate = None
        self._raw_parameters = None
        self.logger.debug("Generated request {} with sufficient capacities at the substrate".format(name))
        return req

    def _generate_request_graph(self, name):
        if self._raw_parameters["probability"] > 0.0:
            req = datamodel.Request(name)
        else:
            req = datamodel.LinearRequest(name)
        selected_edge_resources = numpy.random.exponential(self.average_request_edge_resources)
        # initialize source & sink:
        source_type = self._next_node_type()
        target_type = self._next_node_type()
        source_location, target_location = self._choose_source_target_locations(source_type, target_type)
        source_resources = selected_edge_resources / self.average_request_edge_resources * self.average_request_node_resources_per_type[source_type]
        req.add_node(ServiceChainGenerator.SOURCE_NODE,
                     source_resources,
                     source_type,
                     allowed_nodes={source_location})

        # generate the main chain:
        number_of_nodes_in_core = random.randint(self._raw_parameters["min_number_of_nodes"],
                                                 self._raw_parameters["max_number_of_nodes"])
        # pick a node_type for every node of the requests
        selected_functions = [self._next_node_type() for _ in range(number_of_nodes_in_core)]
        core_nodes_and_functions = []
        previous_node = ServiceChainGenerator.SOURCE_NODE
        for node_type in selected_functions:
            new_node = str(len(req.get_nodes()) - 1)
            core_nodes_and_functions.append((new_node, node_type))
            node_demand = selected_edge_resources / self.average_request_edge_resources * self.average_request_node_resources_per_type[node_type]
            req.add_node(new_node,
                         node_demand,
                         node_type,
                         allowed_nodes=self._substrate.get_nodes_by_type(node_type))
            req.add_edge(previous_node, new_node, selected_edge_resources)
            edge = (previous_node, new_node)
            previous_node = new_node

        target_demand = selected_edge_resources / self.average_request_edge_resources * self.average_request_node_resources_per_type[target_type]
        req.add_node(ServiceChainGenerator.TARGET_NODE,
                     target_demand,
                     target_type,
                     allowed_nodes={target_location})
        req.add_edge(previous_node, ServiceChainGenerator.TARGET_NODE, selected_edge_resources)

        req.node[ServiceChainGenerator.SOURCE_NODE]["fixed_node"] = True
        req.node[ServiceChainGenerator.TARGET_NODE]["fixed_node"] = True
        # Add random connections:
        for n1, node_type_1 in core_nodes_and_functions:
            for n2, node_type_2 in core_nodes_and_functions:
                if n1 == n2 or (n1, n2) in req.get_edges():
                    continue
                if random.random() < self._raw_parameters["probability"]:
                    req.add_edge(n1, n2, selected_edge_resources)
        return req

    def _choose_source_target_locations(self, source_type, target_type):
        potential_source_nodes = list(self._substrate.get_nodes_by_type(source_type))
        potential_target_nodes = list(self._substrate.get_nodes_by_type(target_type))
        source_location = random.choice(potential_source_nodes)
        target_location = random.choice(potential_target_nodes)
        return source_location, target_location

    def _calculate_average_resource_demands(self):
        connection_probability = self._raw_parameters["probability"]
        min_number_nodes = self._raw_parameters["min_number_of_nodes"]
        max_number_nodes = self._raw_parameters["max_number_of_nodes"]
        number_of_requests = self._raw_parameters["number_of_requests"]
        node_res = self._raw_parameters["node_resource_factor"]
        edge_res_factor = self._raw_parameters["edge_resource_factor"]

        average_number_of_nodes_in_core = ((min_number_nodes + max_number_nodes) / 2.0)

        expected_number_of_request_nodes = (float(number_of_requests) * (2 + average_number_of_nodes_in_core))  # add 2 for source & sink
        expected_number_of_request_edges = (float(number_of_requests) * (
            # edges of the main service chain:
                (average_number_of_nodes_in_core - 1 + 2) +
                # edges from random connections:
                connection_probability * (average_number_of_nodes_in_core * (average_number_of_nodes_in_core - 2) + 1)
        ))

        # TODO: this code assumes that all node types are evenly distributed in the request!
        expected_number_of_request_nodes_per_node_type = expected_number_of_request_nodes / float(len(self._node_types))
        self.average_request_node_resources_per_type = {}
        for node_type in self._node_types:
            self.average_request_node_resources_per_type[node_type] = node_res * (self._substrate.get_total_node_resources(node_type) / expected_number_of_request_nodes_per_node_type)

        self.average_request_edge_resources = (1.0 / edge_res_factor) * self._substrate.get_total_edge_resources() / expected_number_of_request_edges


class ExponentialRequestGenerator(AbstractRequestGenerator):
    """
    Generate requests, where the number of nodes is sampled from an exponential distribution.
    Edges are connected at random according to the "probability" parameter.

    Warning: Request graphs may have multiple components.
    """

    EXPECTED_PARAMETERS = [
        "number_of_requests",  # used for estimating average resource demand
        "min_number_of_nodes", "max_number_of_nodes",
        "probability",
        "node_resource_factor",
        "edge_resource_factor",
        "normalize"
    ]

    def __init__(self, logger=None):
        super(ExponentialRequestGenerator, self).__init__(logger)
        self._raw_parameters = None

    def generate_request(self,
                         name,
                         raw_parameters,
                         substrate):
        self._node_types = list(substrate.types)
        self._substrate = substrate
        self._raw_parameters = raw_parameters

        self.logger.debug("Generating request {}".format(name))
        req = self._generate_request_graph(name)
        while not self.verify_substrate_has_sufficient_capacity(req, substrate):
            req = self._generate_request_graph(name)

        self._substrate = None
        self._raw_parameters = None
        self._scenario_parameters_have_changed = True
        return req

    def _generate_request_graph(self, name):
        req = datamodel.Request(name)

        connection_probability = self._raw_parameters["probability"]
        node_res = self._raw_parameters["node_resource_factor"]
        edge_res_factor = self._raw_parameters["edge_resource_factor"]
        min_number_nodes = self._raw_parameters["min_number_of_nodes"]
        max_number_nodes = self._raw_parameters["max_number_of_nodes"]
        number_of_requests = self._raw_parameters["number_of_requests"]

        average_number_of_nodes = ((max_number_nodes + min_number_nodes) / 2.0)

        expected_number_of_request_nodes_per_node_type = (float(number_of_requests) * average_number_of_nodes) / float(len(self._node_types))
        expected_number_of_request_edges = float(number_of_requests) * (average_number_of_nodes * (average_number_of_nodes - 1) * connection_probability)

        average_request_node_resources = {}
        for node_type in self._node_types:
            average_request_node_resources[node_type] = node_res * (self._substrate.get_total_node_resources(node_type) / float(expected_number_of_request_nodes_per_node_type))

        average_request_edge_resources = (1.0 / edge_res_factor) * self._substrate.get_total_edge_resources() / float(expected_number_of_request_edges)

        selected_edge_resources = numpy.random.exponential(average_request_edge_resources)

        # create nodes
        number_of_nodes = random.randint(min_number_nodes, max_number_nodes)
        for i in xrange(1, number_of_nodes + 1):
            node_type = self._next_node_type()
            node_demand = selected_edge_resources / average_request_edge_resources * average_request_node_resources[node_type]
            req.add_node(str(i), node_demand, node_type, allowed_nodes=self._substrate.get_nodes_by_type(node_type))

        # create edges
        for i in req.nodes:
            for j in req.nodes:
                if i == j:
                    continue
                if random.random() <= connection_probability:
                    req.add_edge(i, j, selected_edge_resources)
        return req


class UniformRequestGenerator(AbstractRequestGenerator):
    """
    Generate requests, where the number of nodes is sampled from a uniform distribution.
    Edges are connected at random according to the "probability" parameter.

    Warning: Request graphs may have multiple components.
    """
    EXPECTED_PARAMETERS = [
        "number_of_requests",  # used for estimating average resource demand
        "min_number_of_nodes", "max_number_of_nodes",
        "probability",
        "variability",
        "node_resource_factor",
        "edge_resource_factor",
        "normalize"
    ]

    def __init__(self, logger=None):
        super(UniformRequestGenerator, self).__init__(logger)

    def generate_request(self, name, raw_parameters, substrate):
        self._node_types = list(substrate.types)
        self._substrate = substrate
        self._raw_parameters = raw_parameters

        req = self._generate_request_graph(name)
        while not self.verify_substrate_has_sufficient_capacity(req, substrate):
            req = self._generate_request_graph(name)

        self._scenario_parameters_have_changed = True
        self._raw_parameters = None
        self._substrate = None
        return req

    def _generate_request_graph(self, name):
        req = datamodel.Request(name)
        number_of_substrate_nodes = self._substrate.get_number_of_nodes()
        number_of_substrate_edges = self._substrate.get_number_of_edges()

        variability = self._raw_parameters["variability"]
        connection_probability = self._raw_parameters["probability"]
        # potential_nodes_factor = self._scenario_parameters["potential_nodes_factor"]
        node_res_factor = self._raw_parameters["node_resource_factor"]
        edge_res_factor = self._raw_parameters["edge_resource_factor"]
        min_number_nodes = self._raw_parameters["min_number_of_nodes"]
        max_number_nodes = self._raw_parameters["max_number_of_nodes"]
        number_of_requests = self._raw_parameters["number_of_requests"]

        average_number_of_nodes_per_request = (max_number_nodes + min_number_nodes) / 2.0
        expected_number_of_request_nodes_per_node_type = float(number_of_requests) * average_number_of_nodes_per_request / float(len(self._node_types))

        min_request_node_demand_per_type = {}
        max_request_node_demand_per_type = {}
        for node_type in self._node_types:
            average_request_node_resources = node_res_factor * (self._substrate.get_total_node_resources(node_type) / expected_number_of_request_nodes_per_node_type)
            # apply variability:
            min_request_node_demand_per_type[node_type] = (1.0 - variability) * average_request_node_resources
            max_available_cap = max(self._substrate.get_node_type_capacity(u, node_type) for u in self._substrate.get_nodes_by_type(node_type))
            if min_request_node_demand_per_type[node_type] > max_available_cap:
                msg = "Parameters will always result in infeasible request due to demand for node type {}:\n    Min capacity: {} Max available: {}\n    {}".format(
                    node_type,
                    min_request_node_demand_per_type[node_type],
                    max_available_cap,
                    self._raw_parameters
                )
                raise ScenarioParameterError(msg)
            max_request_node_demand_per_type[node_type] = (1.0 + variability) * average_request_node_resources

        expected_number_of_request_edges = float(number_of_requests) * (average_number_of_nodes_per_request * (average_number_of_nodes_per_request - 1) * connection_probability)
        average_edge_resources = (1.0 / edge_res_factor) * self._substrate.get_total_edge_resources() / expected_number_of_request_edges

        min_request_edge_demand = (1.0 - variability) * average_edge_resources
        max_available_cap = max(self._substrate.get_edge_capacity(uv) for uv in self._substrate.edges)
        if min_request_edge_demand > max_available_cap:
            msg = "Parameters will always result in infeasible request due to edge demand: Min capacity: {} Max available: {}\n    {}".format(
                min_request_edge_demand,
                max_available_cap,
                self._raw_parameters
            )
            raise ScenarioParameterError(msg)
        max_request_edge_demand = (1.0 + variability) * average_edge_resources

        selected_number_of_nodes = random.randint(min_number_nodes, max_number_nodes)

        for node in xrange(1, selected_number_of_nodes + 1):
            node_type = self._next_node_type()
            node_demand = random.uniform(min_request_node_demand_per_type[node_type], max_request_node_demand_per_type[node_type])
            req.add_node(str(node), node_demand, node_type, allowed_nodes=self._substrate.get_nodes_by_type(node_type))

        for node in req.nodes:
            for otherNode in req.nodes:
                if node == otherNode:
                    continue
                if random.random() <= connection_probability:
                    edge_capacity = random.uniform(min_request_edge_demand, max_request_edge_demand)
                    req.add_edge(node, otherNode, edge_capacity)
        return req


_CactusSubTree = namedtuple("_CactusSubTree", "root nodes")


class CactusRequestGenerator(AbstractRequestGenerator):
    """
    Generate request topologies with the cactus graph property.

    First, a random tree is generated according to the "layers" and
    "branching_distribution" scenario parameters.
    
    * ``layers``: an integer giving the depth of the tree
    * ``branching_distribution``: a list of floating point numbers,
      which specifies a probability distribution for the number of children:
      If branching_distribution[i] = p_i, then each node in the tree has
      probability p_i of having i many children.
      The elements of the branching_distribution list must add up to 1.0.

      Warning: if a non-zero value is assigned to the first list element of the branching
      distribution, the tree depth may be reduced and the request graph may degenerate to a single node

    After generating the tree, edges are randomly added to the request graph in a way that
    maintains the cactus property according to the "max_cycles" and "probability" parameters:
    
    * ``max_cycles``: An integer giving a hard upper limit on the number of cycles that may be included
    * ``probability``: The algorithm repeatedly picks two random nodes from a subtree of the graph
      and draws an edge between them according to this parameter.

    To estimate the number of nodes/edges (i.e. the resource footprint), a number of
    graphs is generated according to the "iterations" parameter.
    """
    ROOT = "root"
    EXPECTED_PARAMETERS = [
        "probability",
        "number_of_requests",
        "node_resource_factor",
        "edge_resource_factor",
        "min_number_of_nodes", "max_number_of_nodes",
        "branching_distribution",
        "layers",
        "max_cycles",
        "iterations",
        "fix_root_mapping",
        "fix_leaf_mapping",
        "normalize",
        "arbitrary_edge_orientations"
    ]

    def __init__(self, logger=None):
        super(CactusRequestGenerator, self).__init__(logger)
        self._function_placement_restrictions = None
        self._raw_parameters = None
        self._substrate = None
        self._node_demand_by_type = None
        self._edge_demand = None
        self._generation_attempts = None
        self._advanced_inspection_information = None

    def generate_request(self,
                         name,
                         raw_parameters,
                         substrate):
        self._node_types = list(substrate.types)
        self._raw_parameters = raw_parameters
        self._substrate = substrate
        if self._scenario_parameters_have_changed:  # this operation may be quite expensive => only do it when the values are outdated
            self._calculate_average_resource_demands()
        req = None
        is_feasible = False
        self._generation_attempts = 0
        while not is_feasible:
            req = self._generate_request_graph(name)
            is_feasible = self.verify_substrate_has_sufficient_capacity(req, substrate)
            self._generation_attempts += 1
            if self._generation_attempts > 10**7:
                self._abort()
        self.logger.debug("Generated cactus request {} with sufficient capacities at the substrate".format(name))
        self._scenario_parameters_have_changed = True  # assume that scenario_parameters will change before next call to generate_request
        return req

    def _generate_request_graph(self, name):
        self._select_node_edge_resources()
        req = self._generate_tree_with_correct_size(name)
        self._add_cactus_edges(req)
        if self._raw_parameters["arbitrary_edge_orientations"]:
            reoriented_req = datamodel.Request(req.name)
            for node in req.nodes:
                reoriented_req.add_node(node,
                                        req.node[node]['demand'],
                                        req.node[node]['type'],
                                        req.node[node]['allowed_nodes'])
            for (u, v) in req.edges:
                u_prime, v_prime = u, v
                if random.random() < 0.5:
                    u_prime, v_prime = v, u

                reoriented_req.add_edge(u_prime, v_prime,
                                        req.edge[(u, v)]['demand'],
                                        req.edge[(u, v)]['allowed_edges'])
            return reoriented_req
        return req

    def _generate_tree_with_correct_size(self, name):
        has_correct_size = False
        if self._generation_attempts is None:
            self._generation_attempts = 0
        req = None
        while not has_correct_size:
            req = self._generate_tree(name)
            has_correct_size = self._raw_parameters["min_number_of_nodes"] <= len(req.nodes) <= self._raw_parameters["max_number_of_nodes"]
            if self._advanced_inspection_information is not None:
                self._advanced_inspection_information.generation_tries_overall += 1
                if not has_correct_size:
                    self._advanced_inspection_information.generation_tries_failed += 1
            self._generation_attempts += 1
            if self._generation_attempts > 10**7:
                self._abort()
        return req

    def _generate_tree(self, name):
        req = datamodel.Request(name)
        layer = 0
        fixed_nodes = []
        root_type = self._next_node_type()
        root_id = "{}_{}".format(CactusRequestGenerator.ROOT, req.name)
        req.graph["root"] = root_id
        allowed_nodes = self._substrate.get_nodes_by_type(root_type)
        demand = self._node_demand_by_type[root_type]
        self._add_node_to_request(req, root_id, demand, root_type, allowed_nodes, None, layer)
        previous_layer = [root_id]
        if self._raw_parameters["fix_root_mapping"]:
            fixed_nodes.append(root_id)
        for layer in xrange(1, self._raw_parameters["layers"] + 1):
            current_layer = []
            while previous_layer:
                parent_node = previous_layer.pop()
                number_of_children = self._pick_number_of_children()
                while layer <= 1 and number_of_children < 1:
                    number_of_children = self._pick_number_of_children()
                if number_of_children == 0 and self._raw_parameters["fix_leaf_mapping"]:
                    fixed_nodes.append(parent_node)
                for i in xrange(1, number_of_children + 1):
                    ntype = self._next_node_type()
                    child_node = self._get_node_name(len(req.nodes) + 1, parent_node, layer, ntype, req)
                    current_layer.append(child_node)
                    allowed_nodes = self._substrate.get_nodes_by_type(ntype)
                    demand = self._node_demand_by_type[ntype]
                    self._add_node_to_request(req, child_node, demand, ntype, allowed_nodes, parent_node, layer)
                    if layer == self._raw_parameters["layers"] and self._raw_parameters["fix_leaf_mapping"]:
                        fixed_nodes.append(child_node)
                    req.add_edge(parent_node, child_node, self._edge_demand)

            previous_layer = current_layer
        self._fix_nodes(req, fixed_nodes)
        return req

    def _fix_nodes(self, req, nodes):
        for i in nodes:
            req.node[i]["fixed_node"] = True  # in case the node placement restrictions are overwritten later
            node_type = self._substrate.get_nodes_by_type(req.node[i]["type"])
            allowed_nodes = random.sample(node_type, 1)
            self.logger.debug("{}: Fixing node {} -> {}".format(req.name, i, allowed_nodes))
            req.node[i]["allowed_nodes"] = allowed_nodes

    def _add_node_to_request(self, req, node, demand, ntype, allowed, parent, layer):
        req.add_node(node, demand, ntype, allowed_nodes=allowed)
        req.node[node]["parent"] = parent
        req.node[node]["layer"] = layer

    def _add_cactus_edges(self, req):
        sub_trees = [_CactusSubTree(req.graph["root"], list(req.nodes))]
        cycles = 0
        forbidden_edges = set()
        if len(req.nodes) <= 2:
            return
        edges_on_cycle = 0

        while sub_trees and (cycles < self._raw_parameters["max_cycles"]):
            cycles += 1
            # choose a subtree
            subtree = random.choice(sub_trees)

            # Choose a non-adjacent random pair of nodes within this subtree
            i = random.choice(subtree.nodes)
            j = random.choice(subtree.nodes)
            while i == j or (i in req.get_out_neighbors(j)) or (j in req.get_out_neighbors(i)):
                i = random.choice(subtree.nodes)
                j = random.choice(subtree.nodes)
            if req.node[i]["layer"] > req.node[j]["layer"]:
                i, j = j, i  # make edges always point down the tree

            edge_was_added = False
            # decide to add new edge or not
            if random.random() < self._raw_parameters["probability"]:
                req.add_edge(i, j, self._edge_demand)
                edge_was_added = True

            # forbid any edges on the cycle to reduce the subtree list, regardless of whether edge was added to request
            path_i = CactusRequestGenerator._path_to_root(req, i, subtree.root)
            path_j = CactusRequestGenerator._path_to_root(req, j, subtree.root)
            cycle_edges = path_i.symmetric_difference(path_j)  # only edges on the path to the first common ancestor lie on cycle
            if edge_was_added:
                edges_on_cycle += len(cycle_edges)
            forbidden_edges = forbidden_edges.union(cycle_edges)

            # Update the list of subtrees
            sub_trees = CactusRequestGenerator._list_nontrivial_allowed_subtrees(req, forbidden_edges)

        if self._advanced_inspection_information is not None:
            self._advanced_inspection_information.overall_cycle_edges += edges_on_cycle

    @staticmethod
    def _path_to_root(req, u, root_node):
        result = set()
        while u != root_node:
            parent = req.node[u]["parent"]
            result.add((parent, u))
            u = parent
        return result

    @staticmethod
    def _list_nontrivial_allowed_subtrees(req, edges_on_cycle):
        visited_subtrees = set()
        result = []
        for root_node in sorted(req.nodes, key=lambda i: req.node[i]["layer"]):
            if root_node in visited_subtrees:
                continue
            visited_subtrees.add(root_node)
            subtree = CactusRequestGenerator._get_subtree_nodes_under_node(req, root_node, edges_on_cycle)
            visited_subtrees |= set(subtree)
            if len(subtree) > 2:
                result.append(_CactusSubTree(root_node, subtree))
        return result

    @staticmethod
    def _get_subtree_nodes_under_node(req, root, removed_edges):
        stack = [root]
        subtree_nodes = []
        forbidden_nodes = set(itertools.chain(*removed_edges))
        while stack:
            u = stack.pop()
            subtree_nodes.append(u)
            for w in req.get_out_neighbors(u):
                if u not in forbidden_nodes or w not in forbidden_nodes:
                    stack.append(w)
        return subtree_nodes

    def _pick_number_of_children(self):
        draw = random.random()
        number_of_children = None
        cumulative = 0.0
        for branch_number, probability in enumerate(self._raw_parameters["branching_distribution"]):
            cumulative += probability
            if draw < cumulative:
                number_of_children = branch_number
                break
        if number_of_children is None:
            raise RequestGenerationError("No branching number could be determined!")
        return number_of_children

    def _get_node_name(self, node_id, parent_node, layer, ntype, req):
        if parent_node == req.graph["root"]:
            parent_id = "0"
        else:
            parent_id = parent_node.split("_")[0][1:]
        return "n{id}_{type}_p{parent}_l{layer}_{req}".format(
            id=node_id, type=ntype, parent=parent_id, layer=layer, req=req.name
        )

    def _calculate_average_resource_demands(self):
        number_of_requests = self._raw_parameters["number_of_requests"]

        self._expected_number_of_request_nodes_per_type, self._expected_number_of_request_edges = self._empirical_number_of_nodes_edges()
        self._expected_number_of_request_nodes_per_type *= number_of_requests / len(self._node_types)
        self._expected_number_of_request_edges *= number_of_requests
        node_res = self._raw_parameters["node_resource_factor"]
        edge_res_factor = self._raw_parameters["edge_resource_factor"]
        self.average_request_edge_resources = (1.0 / edge_res_factor) * self._substrate.get_total_edge_resources() / self._expected_number_of_request_edges
        self.average_request_node_resources_per_type = {}
        for ntype in self._node_types:
            self.average_request_node_resources_per_type[ntype] = node_res * (self._substrate.get_total_node_resources(ntype) / self._expected_number_of_request_nodes_per_type)

    def _select_node_edge_resources(self):
        self._edge_demand = numpy.random.exponential(self.average_request_edge_resources)
        self._node_demand_by_type = {}
        for ntype in self._node_types:
            self._node_demand_by_type[ntype] = self._edge_demand / self.average_request_edge_resources * self.average_request_node_resources_per_type[ntype]
            # print ("Selected resources: Edges: {:8.2f}    Nodes by type: {}   ".format(self._edge_demand, self._node_demand_by_type))

    def expected_number_of_nodes_in_tree(self):
        nodes_at_layer = {}
        expected_number_of_children = sum(i * p for i, p in enumerate(self._raw_parameters["branching_distribution"]))

        def expected_nodes_in_sublayers(layer):
            if layer in nodes_at_layer:
                return nodes_at_layer[layer]
            result = 1
            if layer != self._raw_parameters["layers"]:
                result += expected_number_of_children * expected_nodes_in_sublayers(layer + 1)
            nodes_at_layer[layer] = result
            return result

        node_count = expected_nodes_in_sublayers(0)
        return node_count

    def _empirical_number_of_nodes_edges(self):
        total_nodes = 0
        total_edges = 0
        self._node_demand_by_type = {nt: 0.0 for nt in self._node_types}
        self._edge_demand = 0.0
        r_state = random.getstate()
        iterations = self._raw_parameters["iterations"]
        for i in xrange(iterations):
            req = self._generate_tree_with_correct_size("test")
            self._add_cactus_edges(req)
            total_nodes += len(req.nodes)
            total_edges += len(req.edges)
        random.setstate(r_state)
        total_nodes /= float(iterations)
        total_edges /= float(iterations)
        self.logger.info("Expecting {} nodes, {} edges".format(total_nodes, total_edges))
        return total_nodes, total_edges

    class AdvancedInspectionResult(object):

        def __init__(self):
            self.generation_tries_overall = 0
            self.generation_tries_failed = 0
            self.nodes_generated = 0
            self.edges_generated = 0
            self.overall_cycle_edges = 0
            self.node_edge_comination = []
            self.generated_cycles = []

    def advanced_empirical_number_of_nodes_edges(self, raw_parameters, substrate, iterations):
        self._advanced_inspection_information = self.AdvancedInspectionResult()

        raw_parameters["iterations"] = 1

        for i in xrange(iterations):

            if i % 100000 == 0 and i > 0:
                print("{} of {} iterations done.".format(i, iterations))

            req = self.generate_request(name="test",
                                        raw_parameters=raw_parameters,
                                        substrate=substrate)

            number_nodes = len(req.nodes)
            number_edges = len(req.edges)
            cycles = number_edges - number_nodes + 1
            if cycles == 0:
                print "NO CYCLES!"
            self._advanced_inspection_information.generated_cycles.append(cycles)

            self._advanced_inspection_information.node_edge_comination.append((number_nodes, number_edges))

            self._advanced_inspection_information.nodes_generated += number_nodes
            self._advanced_inspection_information.edges_generated += number_edges

        return self._advanced_inspection_information

    def _abort(self):
        raise RequestGenerationError(
            "Could not generate a Cactus request after {} attempts!\n{}".format(self._generation_attempts, self._raw_parameters)
        )


class TreewidthRequestGenerator(AbstractRequestGenerator):
    """
    Generate request topologies of bounded treewidth using a mandatory UndirectedGraphStorage.
    Specifically, for graphs of treewidth 1 simple trees are generated while for higher treewidths the
    graphs from the UndirectedGraphStorage are used.

    To specify an UndirectedGraphStorage, a pickle has to be given in the yaml-file as a data manager like
    data_managers:
      UndirectedGraphStorage: <location_of_pickle>

    """
    EXPECTED_PARAMETERS = [
        "number_of_requests",
        "treewidth",
        "min_number_of_nodes",
        "max_number_of_nodes",
        "node_resource_factor",
        "edge_resource_factor",
        "normalize"
    ]

    def __init__(self, logger=None):
        super(TreewidthRequestGenerator, self).__init__(logger)
        self._raw_parameters = None
        self._substrate = None
        self._node_demand_by_type = None
        self._edge_demand = None
        self.DEBUG_MODE = False #set to True to enable checking tree decompositions during scenario generation

    def generate_request(self,
                         name,
                         raw_parameters,
                         substrate):
        self._node_types = list(substrate.types)
        self._raw_parameters = raw_parameters
        self._substrate = substrate
        self._treewidth = self._raw_parameters["treewidth"]
        self._average_edge_numbers_of_treewidth = None
        self._undirected_graph_storage = None
        self._number_of_requests = self._raw_parameters["number_of_requests"]
        self._min_number_nodes = int(self._raw_parameters["min_number_of_nodes"])
        self._max_number_nodes = int(self._raw_parameters["max_number_of_nodes"])
        self._number_of_nodes = None

        self._generation_attempts = 0

        if self._treewidth > 1:
            graph_storage, graph_storage_lock = self._data_manager_dict["UndirectedGraphStorage"]
            self._undirected_graph_storage = graph_storage
            self._undirected_graph_storage_lock = graph_storage_lock
            if self._average_edge_numbers_of_treewidth is None:
                self._average_edge_numbers_of_treewidth = {}
            if self._treewidth not in self._average_edge_numbers_of_treewidth.keys():
                with self._undirected_graph_storage_lock:
                    self._average_edge_numbers_of_treewidth[
                        self._treewidth] = self._undirected_graph_storage.get_average_number_of_edges_for_parameter(
                        self._treewidth)
        self._calculate_average_resource_demands()
        req = None
        is_feasible = False


        while not is_feasible:
            req = self._generate_request_graph(name)
            is_feasible = self.verify_substrate_has_sufficient_capacity(req, substrate)
            self._generation_attempts += 1
            if self._generation_attempts > 10**7:
                self._abort()
        self._scenario_parameters_have_changed = True  # assume that scenario_parameters will change before next call to generate_request
        return req

    def _calculate_average_resource_demands(self):

        self._expected_number_of_request_nodes_per_type = float(self._number_of_requests * (self._min_number_nodes + self._max_number_nodes) / 2.0) / len(self._node_types)
        self._expected_number_of_request_edges = None
        if self._treewidth == 1:
            self._expected_number_of_request_edges = (self._expected_number_of_request_nodes_per_type * len(self._node_types)) - 1
        else:
            if self._average_edge_numbers_of_treewidth is None:
                self._average_edge_numbers_of_treewidth = {}
            if self._treewidth not in self._average_edge_numbers_of_treewidth.keys():
                with self._undirected_graph_storage_lock:
                    self._average_edge_numbers_of_treewidth[self._treewidth] = self._undirected_graph_storage.get_average_number_of_edges_for_parameter(self._treewidth)
            self._expected_number_of_request_edges = 0

            number_of_nodes_count = 0
            for number_of_nodes in range(self._min_number_nodes, self._max_number_nodes + 1):
                if number_of_nodes in self._average_edge_numbers_of_treewidth[self._treewidth]:
                    self._expected_number_of_request_edges += self._average_edge_numbers_of_treewidth[self._treewidth][number_of_nodes]
                    number_of_nodes_count += 1
                else:
                    self.logger.warning(
                        "The undirected graph storage does not contain graphs for treewidth {} having exactly {} nodes.".format(
                            self._treewidth, number_of_nodes))
            self._expected_number_of_request_edges = float(self._number_of_requests) * float(self._expected_number_of_request_edges) / (float(number_of_nodes_count))

        node_res = self._raw_parameters["node_resource_factor"]
        edge_res_factor = self._raw_parameters["edge_resource_factor"]
        self.average_request_edge_resources = (
                                                      1.0 / edge_res_factor) * self._substrate.get_total_edge_resources() / self._expected_number_of_request_edges
        self.average_request_node_resources_per_type = {}
        for ntype in self._node_types:
            self.average_request_node_resources_per_type[ntype] = node_res * (self._substrate.get_total_node_resources(
                ntype) / self._expected_number_of_request_nodes_per_type)

    def _select_node_edge_resources(self):
        self._edge_demand = numpy.random.exponential(self.average_request_edge_resources)
        self._node_demand_by_type = {}
        for ntype in self._node_types:
            self._node_demand_by_type[ntype] = self._edge_demand / self.average_request_edge_resources * \
                                               self.average_request_node_resources_per_type[ntype]

    def _select_number_of_nodes(self):
        while True:
            # round as long as there exists a graph with this number of nodes
            self._number_of_nodes = random.randint(self._min_number_nodes, self._max_number_nodes)
            if self._treewidth == 1 or self._number_of_nodes in self._average_edge_numbers_of_treewidth[self._treewidth].keys():
                break
            else:
                self.logger.warning("The undirected graph storage does not contain graphs for treewidth {} having exactly {} nodes.".format(self._treewidth, self._number_of_nodes))

    def _generate_edge_list_representation(self):
        if self._treewidth == 1:
            return self._generate_edge_representation_of_tree(self._number_of_nodes)
        else:
            result = None
            with self._undirected_graph_storage_lock:
                result = self._undirected_graph_storage.get_random_graph_as_edge_list_representation(self._treewidth, self._number_of_nodes)
            return result

    def _generate_edge_representation_of_tree(self, number_of_nodes):
        # first generate all potential edges
        all_edges = [(str(i), str(j)) for i in range(1, number_of_nodes + 1) for j in range(i + 1, number_of_nodes + 1)]
        # shuffle them
        random.shuffle(all_edges)
        # use edges as long as these do not close cycles; to achieve that we use again the concept of connected components
        # initially each node only reaches itself, while an edge connecting two nodes leads to the merging of
        # the respective two connected components

        node_to_connected_component_id = {str(i): i for i in range(1, number_of_nodes + 1)}
        connected_component_id_to_nodes = {i: [str(i)] for i in range(1, number_of_nodes + 1)}
        result = []
        current_edge_index = 0
        while len(connected_component_id_to_nodes.keys()) > 1:
            # check if adding edge would violate tree property
            i, j = all_edges[current_edge_index]
            component_i = node_to_connected_component_id[i]
            component_j = node_to_connected_component_id[j]
            if component_i == component_j:
                # do not add edge
                pass
            else:
                # add edge
                result.append((i, j))
                # merge
                connected_component_id_to_nodes[component_i].extend(connected_component_id_to_nodes[component_j])
                for node in connected_component_id_to_nodes[component_j]:
                    node_to_connected_component_id[node] = component_i
                del connected_component_id_to_nodes[component_j]
            # consider next edge
            current_edge_index += 1
        return result

    def _generate_request_graph(self, name):

        def _get_node_name(node):
            return name + "_" + node

        self._select_number_of_nodes()

        self._select_node_edge_resources()
        req_edge_representation = self._generate_edge_list_representation()
        nodes = datamodel.get_nodes_of_edge_list_representation(req_edge_representation)

        req = datamodel.Request(name)

        for node in nodes:
            # select node type:
            ntype = self._next_node_type()
            allowed_nodes = self._substrate.get_nodes_by_type(ntype)

            req.add_node(_get_node_name(node),
                         self._node_demand_by_type[ntype],
                         ntype,
                         allowed_nodes)

        for edge in req_edge_representation:
            i, j = edge
            # draw random orientation of edge:
            actual_edge = (_get_node_name(i), _get_node_name(j))
            if random.random() <= 0.5:
                actual_edge = (_get_node_name(j), _get_node_name(i))
            i, j = actual_edge
            req.add_edge(i, j, self._edge_demand)

        if self.DEBUG_MODE:
            from vnep_approx import treewidth_model
            # check connectdness of request graph
            tmp_edges = list(req.edges)
            tmp_undir_req_graph = datamodel.get_undirected_graph_from_edge_representation(tmp_edges)
            assert tmp_undir_req_graph.check_connectedness()
            td_comp = treewidth_model.TreeDecompositionComputation(tmp_undir_req_graph)
            tree_decomp = td_comp.compute_tree_decomposition()
            assert tree_decomp.width == self._treewidth
            td_comp = treewidth_model.TreeDecompositionComputation(req)
            tree_decomp = td_comp.compute_tree_decomposition()
            assert tree_decomp.width == self._treewidth
            sntd = treewidth_model.SmallSemiNiceTDArb(tree_decomp, req)
            self.logger.info("SUCCESSFULLY PASSED TESTS [DEBUG_MODE=TRUE]")
            assert len(req.nodes) == self._number_of_nodes

        return req

    def _abort(self):
        raise RequestGenerationError(
            "Could not generate a Cactus request after {} attempts!\n{}".format(self._generation_attempts, self._raw_parameters)
        )


class AbstractProfitCalculator(ScenariogenerationTask):
    """Base class for the profit generation task."""

    def __init__(self, logger):
        super(AbstractProfitCalculator, self).__init__(logger)

    def apply(self, scenario_parameters, scenario):
        class_raw_parameters_dict = scenario_parameters[PROFIT_CALCULATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            valid_class_str = ", ".join(str(c) for c in class_raw_parameters_dict.keys())
            raise ScenarioGeneratorError("{class_name} is not a valid profit calculation tasks (expected one of {valid_classes})".format(
                class_name=class_name, valid_classes=valid_class_str
            ))
        raw_parameters = class_raw_parameters_dict[class_name]
        self.generate_and_apply_profits(scenario, raw_parameters)

    def generate_and_apply_profits(self, scenario, raw_parameters):
        raise NotImplementedError("This is an abstract class! Use one of the implementations defined below.")


class RandomEmbeddingProfitCalculator(AbstractProfitCalculator):
    """
    Calculate profits for all new requests in a scenario.

    The profit for each request is the average cost of random embeddings
    in the empty substrate, multiplied by the "profit_factor" parameter.

    The "iterations" parameter defines the number of random embeddings.
    """

    EXPECTED_PARAMETERS = [
        "profit_factor",
        "iterations"
    ]

    def __init__(self, logger=None):
        super(RandomEmbeddingProfitCalculator, self).__init__(logger)
        self._scenario = None
        self._iterations = None

    def generate_and_apply_profits(self, scenario, raw_parameters):
        self._scenario = scenario
        self._iterations = raw_parameters["iterations"]
        self.logger.info("Calculating vnet profits based on random embedding ({} iterations)".format(self._iterations))

        if not self._scenario.substrate.shortest_paths_costs:
            self._scenario.substrate.initialize_shortest_paths_costs()

        for req in self._scenario.requests:
            start_time = time.clock()
            cost = self._get_average_cost_from_embedding_graph_randomly(
                req, self._scenario.substrate.shortest_paths_costs
            )
            req.profit = -cost * raw_parameters["profit_factor"]
            end_time = time.clock()
            req.graph["profit_calculation_time"] = end_time - start_time
            self.logger.debug("\t{}\t{}".format(req.name, req.profit))

        self._iterations = None
        self._scenario = None

    def _get_average_cost_from_embedding_graph_randomly(self, req, shortest_paths):
        costs = [0 for _ in xrange(self._iterations)]
        average_factor = 1.0 / self._iterations
        for i in xrange(self._iterations):
            mapped_node = dict()
            costs[i] = 0.0
            for node in req.get_nodes():
                restrictions = req.get_allowed_nodes(node)
                if restrictions is None or len(restrictions) == 0:
                    restrictions = self._scenario.substrate.get_nodes()
                snode = random.choice(tuple(restrictions))
                mapped_node[node] = snode
                # mapping costs are negative to be consistent with mip calculation:
                costs[i] -= self._scenario.substrate.get_node_type_cost(snode, req.node[node]["type"]) * req.node[node]["demand"] * average_factor
            for edge in req.get_edges():
                n1, n2 = edge
                sn1 = mapped_node[n1]
                sn2 = mapped_node[n2]
                # mapping costs are negative to be consistent with mip calculation:
                costs[i] -= shortest_paths[sn1][sn2] * req.edge[edge]["demand"] * average_factor
        return sum(costs)


class OptimalEmbeddingProfitCalculator(AbstractProfitCalculator):
    """
    Calculate profits for all new requests in a scenario.

    The profit for each request is the cost of its optimal embedding
    in the empty substrate, multiplied by the profit_factor parameter.
    """

    EXPECTED_PARAMETERS = [
        "profit_factor",
        "timelimit",
    ]

    def __init__(self, logger=None):
        super(OptimalEmbeddingProfitCalculator, self).__init__(logger)
        self._scenario = None

    def generate_and_apply_profits(self, scenario, raw_parameters):
        self._raw_parameters = raw_parameters
        self._scenario = scenario

        self.logger.info("Calculating vnet profits based on individual (optimal) min-cost embedding.".format())
        cost_list = [self._solve_only_one_vnet_optimally(req)
                     for req in self._scenario.requests]
        self._apply_embedding_cost_as_request_profit(cost_list, raw_parameters)

    def _apply_embedding_cost_as_request_profit(self, embedding_cost, scenario_parameters):
        self.logger.info("Applying vnet profits to scenario {}".format(self._scenario))
        for req, cost in itertools.izip(self._scenario.requests, embedding_cost):
            req.profit = cost * scenario_parameters["profit_factor"]
            self.logger.debug("\tname: {}\tprofit: {}".format(req.name, req.profit))

    def _make_sub_scenario_containing_only(self, request):
        copied_request = copy.deepcopy(request)
        copied_request.setProfit(0.0)
        return datamodel.Scenario("profit_calculation_{}".format(request.name),
                                  self._scenario.substrate,
                                  {copied_request.name: copied_request})

    def _solve_only_one_vnet_optimally(self, req):
        self.logger.debug("Calculating optimal solution for {}".format(req.name))
        copied_request = copy.deepcopy(req)
        copied_request.profit = 0.0
        scenario_copy = copy.deepcopy(self._scenario)
        scenario_copy.requests = [copied_request]
        scenario_copy.objective = datamodel.Objective.MIN_COST

        gurobi_settings = modelcreator.GurobiSettings(mipGap=0.001,
                                                      nodeLimit=None,
                                                      heuristics=None,
                                                      threads=1,
                                                      timelimit=self._raw_parameters["timelimit"])
        mc = mip.ClassicMCFModel(scenario_copy, logger=self.logger)

        mc.init_model_creator()
        mc.model.setParam("OutputFlag", 0)
        mc.apply_gurobi_settings(gurobi_settings)

        solution = mc.compute_integral_solution()
        if solution is None:
            min_embedding_cost = 0.0
        else:
            min_embedding_cost = mc.status.objValue
        req.graph["profit_calculation_time"] = (mc.time_preprocess +
                                                mc.time_optimization +
                                                mc.time_postprocessing)
        # Some cleanup:
        del scenario_copy
        del mc.model
        del mc
        del copied_request
        return min_embedding_cost


class AbstractNodeMappingRestrictionGenerator(ScenariogenerationTask):
    def __init__(self, logger):
        super(AbstractNodeMappingRestrictionGenerator, self).__init__(logger)

    def apply(self, scenario_parameters, scenario):
        class_raw_parameters_dict = scenario_parameters[NODE_PLACEMENT_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise ScenarioGeneratorError("")
        raw_parameters = class_raw_parameters_dict[class_name]
        self.generate_and_apply_restrictions(scenario, raw_parameters)

    def generate_and_apply_restrictions(self, scenario, raw_parameters):
        self.logger.info("{}: Generate node placement restrictions for {}".format(self.__class__.__name__, scenario.name))
        for req in scenario.requests:
            self.generate_restrictions_single_request(req, scenario.substrate, raw_parameters)

    def generate_restrictions_single_request(self, req, substrate, raw_parameters):
        raise NotImplementedError("This is an abstract method! Use one of the implementations defined below.")

    def _number_of_nodes(self, req, node, number_of_substrate_nodes_with_correct_type, raw_parameters):
        number_of_allowed_nodes = int(raw_parameters["potential_nodes_factor"] * number_of_substrate_nodes_with_correct_type)
        number_of_allowed_nodes = max(1, number_of_allowed_nodes)  # at least 1 node should be allowed!
        number_of_allowed_nodes = min(number_of_allowed_nodes, number_of_substrate_nodes_with_correct_type)
        return number_of_allowed_nodes


class UniformEmbeddingRestrictionGenerator(AbstractNodeMappingRestrictionGenerator):
    """
    Generate node placement restrictions by randomly choosing a fixed ratio of the
    substrate nodes with suitable type. At least one substrate node is always chosen.
    If the request node dictionary of a node contains the parameter "number_of_allowed_nodes",
    then this number is used instead.

    Example: Assuming a substrate with 20 nodes, 10 of which can support type "t1" and potential_nodes_factor=0.4.
    Then, each request node of type "t1" is mapped to a random sample of 4 out of the 10 supporting nodes.

    potential_nodes_factor: Floating point number between 0.0 and 1.0.
    """
    EXPECTED_PARAMETERS = [
        "potential_nodes_factor"
    ]

    def __init__(self, logger=None):
        super(UniformEmbeddingRestrictionGenerator, self).__init__(logger)

    def generate_restrictions_single_request(self, req, substrate, raw_parameters):
        self.logger.debug("\tGenerating restrictions for {}".format(req.name))
        for node in req.nodes:
            if "fixed_node" in req.node[node]:
                if req.node[node]["fixed_node"]:
                    self.logger.debug("\t\t{} -> {}".format(node, req.get_allowed_nodes(node)))
                    continue
            allowed_nodes = substrate.get_nodes_by_type(req.node[node]["type"])
            number_of_possible_nodes = len(allowed_nodes)
            number_of_allowed_nodes = self._number_of_nodes(req, node, number_of_possible_nodes, raw_parameters)
            allowed_nodes = random.sample(allowed_nodes, number_of_allowed_nodes)
            allowed_nodes = set(allowed_nodes)
            self.logger.debug("\t\tAllowed substrate nodes for {} -> {}".format(node, allowed_nodes))
            req.set_allowed_nodes(node, allowed_nodes)


class NeighborhoodSearchRestrictionGenerator(AbstractNodeMappingRestrictionGenerator):
    """
    Generate node placement restrictions by traversing the substrate in a breadth-first-search, starting
    at a randomly chosen "center" node. Whenever a substrate node supporting the node type is encountered,
    it is added to the "allowed_nodes" list.

    The number of allowed nodes is sampled from an exponential distribution with the mean "potential_nodes_factor".

    potential_nodes_factor: Floating point number between 0.0 and 1.0.
    """
    EXPECTED_PARAMETERS = [
        "potential_nodes_factor"
    ]

    def __init__(self, logger=None):
        super(NeighborhoodSearchRestrictionGenerator, self).__init__(logger)

    def generate_restrictions_single_request(self, req, substrate, raw_parameters):
        for node in req.nodes:
            if "fixed_node" in req.node[node]:
                if req.node[node]["fixed_node"]:
                    self.logger.debug("\t\t{} -> {}".format(node, req.get_allowed_nodes(node)))
                    continue
            ntype = req.node[node]["type"]
            substrate_nodes_of_correct_type = substrate.get_nodes_by_type(ntype)
            number_of_substrate_nodes = len(substrate_nodes_of_correct_type)

            number_of_allowed_nodes = self._number_of_nodes(req, node, number_of_substrate_nodes, raw_parameters)

            allowed_nodes = set()
            visited_nodes = set()

            center = random.choice(substrate_nodes_of_correct_type)
            allowed_nodes.add(center)
            visited_nodes.add(center)
            nodes_to_explore = deque(neighbor for neighbor in substrate.get_out_neighbors(center))
            while len(allowed_nodes) < number_of_allowed_nodes and nodes_to_explore:
                next_node = nodes_to_explore.pop()
                if next_node in substrate_nodes_of_correct_type:
                    allowed_nodes.add(next_node)
                visited_nodes.add(next_node)
                nodes_to_explore.extendleft(snode for snode in substrate.get_out_neighbors(next_node) if snode not in visited_nodes)

            req.set_allowed_nodes(node, allowed_nodes)
            self.logger.debug("\t\t{} -> {}".format(node, allowed_nodes))


class TopologyZooReader(ScenariogenerationTask):
    '''Casts topology zoo instances as substrates during the generation process

    '''

    # node_cost: [1.0]    #this is a multiplicative factor :)
    EXPECTED_PARAMETERS = ["topology", "node_types", "edge_capacity",
                           "node_cost_factor", "node_capacity", "node_type_distribution",
                           "fog_model_costs"]

    def __init__(self, path=os.path.join(DATA_PATH, "topologyZoo"), logger=None):
        super(TopologyZooReader, self).__init__(logger)
        self.topology_zoo_dir = path
        self._raw_nx_graphs = {}

    def apply(self, scenario_parameters, scenario):
        class_raw_parameters_dict = scenario_parameters[SUBSTRATE_GENERATION_TASK].values()[0]
        class_name = self.__class__.__name__
        if class_name not in class_raw_parameters_dict:
            raise ScenarioGeneratorError("")
        raw_parameters = class_raw_parameters_dict[class_name]
        scenario.substrate = self.read_substrate(raw_parameters)

    def read_substrate(self, raw_parameters):
        filename = raw_parameters["topology"]
        substrate = self.read_from_yaml(raw_parameters)
        substrate.initialize_shortest_paths_costs()
        return substrate

    def read_from_yaml(self, raw_parameters, include_location=False):
        topology = raw_parameters["topology"]
        with open(os.path.abspath(os.path.join(self.topology_zoo_dir, topology + ".yml"))) as f:
            graph_dict = yaml.safe_load(f)

        nodes = graph_dict["nodes"].keys()
        edges = graph_dict["edges"]
        assigned_types = self._assign_node_types(nodes, raw_parameters)

        dists = {}
        # we compute at first the sum of all edge costs
        total_edge_costs = 0
        for edge in edges:
            u, v = edge
            u_lon = graph_dict["nodes"][u]["Longitude"]
            u_lat = graph_dict["nodes"][u]["Latitude"]
            v_lon = graph_dict["nodes"][v]["Longitude"]
            v_lat = graph_dict["nodes"][v]["Latitude"]
            dists[u, v] = haversine(u_lon, u_lat, v_lon, v_lat)
            total_edge_costs += 2 * dists[u, v] * raw_parameters["edge_capacity"]

        # this edge cost shall then equal the sum of all node costs
        # hence we first compute for each type the total available capacity

        total_capacity_per_type = {t: 0.0 for t in raw_parameters["node_types"]}
        for node in nodes:
            for type in assigned_types[node]:
                total_capacity_per_type[type] += raw_parameters["node_capacity"]

        sum_of_capacities = sum(total_capacity_per_type[type] for type in total_capacity_per_type.keys())

        substrate = datamodel.Substrate(topology)

        for node in nodes:
            types = assigned_types[node]
            capacity = {t: raw_parameters["node_capacity"] for t in types}
            if "fog_model_costs" in raw_parameters and bool(raw_parameters["fog_model_costs"]):
                cost = {'universal': 0.0}
            else:
                cost = {t: total_edge_costs / sum_of_capacities for t in types}
            substrate.add_node(node, types, capacity, cost)
            if include_location:
                substrate.node[node]["Longitude"] = graph_dict["nodes"][node]["Longitude"]
                substrate.node[node]["Latitude"] = graph_dict["nodes"][node]["Latitude"]

        average_distance = 0

        for (tail, head), dist in dists.items():
            if "fog_model_costs" in raw_parameters and bool(raw_parameters["fog_model_costs"]):
                cost = 1.0
            else:
                cost = dist
            capacity = raw_parameters["edge_capacity"]
            if raw_parameters.get("include_latencies", False):
                cost, latency = self._get_costs_and_latencies_from_distance(dist)
                average_distance += latency
                substrate.add_edge(tail, head, capacity=capacity, cost=cost, latency=latency)
            else:
                substrate.add_edge(tail, head, capacity=capacity, cost=cost)

        average_distance /= len(substrate.edges)
        substrate.set_average_node_distance(average_distance)

        return substrate


    def _get_costs_and_latencies_from_distance(self, dist):
        """ cost, latency """
        return dist, dist    #latencies and costs are equal and are given in milliseconds TODO: Robin introdued an additional factor of 20, maybe we should check why this was done.


    def _assign_node_types(self, nodes, raw_parameters):

        nodes_per_function = max(1, int(raw_parameters["node_type_distribution"] * len(nodes)))
        nodes_per_function = min(len(nodes), nodes_per_function)
        assigned_types = {u: [] for u in nodes}

        for node_type in raw_parameters["node_types"]:
            for u in random.sample(nodes, nodes_per_function):
                assigned_types[u].append(node_type)
        return assigned_types


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    # 6367 km is the radius of the Earth
    km = 6367 * c
    latency = km / 200
    return latency # in milliseconds


def convert_topology_zoo_gml_to_yml(gml_path, yml_path, consider_disconnected):
    network_files = glob.glob(gml_path  + "/*.gml")

    for net_file in network_files:
        # Extract name of network from file path
        path, filename = os.path.split(net_file)
        network_name = os.path.splitext(filename)[0]


        try:
            print "reading file {}".format(net_file)
            try:
                graph = nx.read_gml(net_file, label="id")
            except Exception as ex:
                if "duplicated" in str(ex) and "multigraph 1" in str(ex):
                    print "Multigraph detected; fixing the problem"
                    with open(net_file, "r") as f:
                        graph_source = f.read()
                    graph_source = graph_source.replace("graph [", "graph [\n  multigraph 1")
                    graph = nx.parse_gml(graph_source, label="id")
                    print "tried to fix it.."



            largest_cc = max(nx.connected_components(graph), key=len)

            if len(largest_cc) < graph.number_of_nodes():
                if consider_disconnected:
                    print "Graph is not connected, considering only the largest connected component with {} nodes".format(len(largest_cc))
                else:
                    print "Graph is not connected, discarding it!"
                    continue


            nodes = largest_cc

            yml_contents = {"nodes": {}, "edges": []}
            for node in nodes:
                data = graph.node[node]
                print node, data
                longitude = 0
                latitude = 0
                if "Longitude" in data:
                    longitude = data["Longitude"]
                else:
                    for neighbor in graph.neighbors(node):
                        if "Longitude" in graph.node[neighbor]:
                            longitude += graph.node[neighbor]["Longitude"]
                        else:
                            print "Could NOT estimate longitude for node {} based on neighbors. Aborting conversion.".format(node)
                            raise RuntimeError("Could not approximate longitude")
                    print "Successfully estimated longitude for node {} based on neighbors".format(node)
                    longitude /= float(len(list(graph.neighbors(node))))


                if "Latitude" in data:
                    latitude = data["Latitude"]
                else:
                    for neighbor in graph.neighbors(node):
                        if "Latitude" in graph.node[neighbor]:
                            latitude += graph.node[neighbor]["Latitude"]
                        else:
                            print "Could NOT estimate latitude for node {} based on neighbors. Aborting conversion.".format(node)
                            raise RuntimeError("Could not approximate latitude")
                    print "Successfully estimated latitude for node {} based on neighbors".format(node)
                    latitude /= float(len(list(graph.neighbors(node))))

                yml_contents["nodes"][str(node)] = {'Longitude': longitude, "Latitude": latitude}
                for data_key, data_value in data.iteritems():
                    if data_key == "Longitude" or data_key == "Latitude":
                        continue
                    else:
                        yml_contents["nodes"][str(node)][unidecode(str(data_key))] = unidecode(str(data_value))


            for u,v, data in graph.edges(data=True):
                if u not in nodes:
                    print("Node {} not contained in graph; edge {} not considered. This should be due to the graph not being connected.".format(u, (u,v)))
                    continue
                if v not in nodes:
                    print(
                        "Node {} not contained in graph; edge {} not considered. This should be due to the graph not being connected.".format(
                            v, (u, v)))
                    continue
                if (str(u), str(v)) in yml_contents["edges"]:
                    print "Edge {} already known (MultiGraph), discarding it".format((str(u), str(v)))
                    continue
                else:
                    yml_contents["edges"].append([str(u),str(v)])

            output_filename = os.path.join(yml_path, network_name + ".yml")
            print "writing {}".format(output_filename)
            with open(output_filename, "w") as output:
                yaml.dump(yml_contents, output)
            print "file {} sucessfully converted to yml file {}! \n\n\n".format(net_file, output_filename)

        except Exception as ex:
            import traceback
            print "conversion of file {} to yml was NOT sucessful! \n\n\n".format(net_file)
            print "non successfull! {}\n\n\n".format(str(ex))
            traceback.print_exc()



def summarize_topology_zoo_graphs(min_number_nodes=10, max_number_nodes=100):
    network_files = glob.glob(os.path.join(DATA_PATH, "topologyZoo/") + "*.yml")
    #network_files = glob.glob(os.path.join(DATA_PATH, "topologyZoo/") + "DeutscheTelekom.yml")

    networks_by_name = {}

    multiple_occuring_networks = {}

    reader = TopologyZooReader()

    print "network_files", network_files

    raw_parameters = {"topology": "UNDEFINED",
                      "node_types": ["universal"],
                      "node_capacity": 100.0,
                      "edge_capacity": 100.0,
                      "node_type_distribution": 1.0}

    for net_file in network_files:
        # Extract name of network from file path
        path, filename = os.path.split(net_file)
        network_name = os.path.splitext(filename)[0]

        raw_parameters["topology"] = network_name

        print "trying to parse {} ".format(network_name)
        graph = reader.read_from_yaml(raw_parameters)

        if not graph.check_connectivity():
            print("graph {} is NOT connected!".format(network_name))
            continue

        networks_by_name[network_name] = graph

        nameWithoutDate = ''.join([i for i in network_name if not i.isdigit()])
        dateInformation = ''.join([i for i in network_name if i.isdigit()])

        if nameWithoutDate != network_name and len(dateInformation) >= 4:
            # there is some sort of year inormation included
            if nameWithoutDate not in multiple_occuring_networks:
                multiple_occuring_networks[nameWithoutDate] = []

            multiple_occuring_networks[nameWithoutDate].append((network_name, dateInformation))

    # select only the most current graphs
    for mNetwork in multiple_occuring_networks.keys():
        listOfNetworks = multiple_occuring_networks[mNetwork]
        bestName = None
        bestDate = None
        for network_name, dateInformation in listOfNetworks:
            if len(dateInformation) < 6:
                dateInformation = dateInformation + "0" * (6 - len(dateInformation))
            if bestDate is None or int(dateInformation) > int(bestDate):
                bestDate = dateInformation
                bestName = network_name

        for network_name, dateInformation in listOfNetworks:
            if network_name != bestName:
                print("deleting {} as it is superseded by {}".format(network_name, bestName))
                del networks_by_name[network_name]

    print networks_by_name
    print multiple_occuring_networks

    # order networks according to increasing complexity
    orderedDictOfNetworks = {}
    for network, graph in networks_by_name.items():
        n = graph.get_number_of_nodes()
        if n < min_number_nodes or n > max_number_nodes:
            print "not considering graph ", network
            continue
        if n not in orderedDictOfNetworks.keys():
            orderedDictOfNetworks[n] = []
        orderedDictOfNetworks[n].append((network, graph))




    numberOfgraphs = 0
    for numberOfNodes in sorted(orderedDictOfNetworks.keys()):
        print("n = {}: {}\n\t{}".format(numberOfNodes, len(orderedDictOfNetworks[numberOfNodes]),[(x,y.get_number_of_edges()) for (x,y) in orderedDictOfNetworks[numberOfNodes]]))
        numberOfgraphs += len(orderedDictOfNetworks[numberOfNodes])

    print("\n" + "-" * 40)
    print("Selected {} graphs.".format(numberOfgraphs))

    print("\nsaving list of selected topologies..\n")





