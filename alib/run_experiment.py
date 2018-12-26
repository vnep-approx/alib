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

import cPickle as pickle
import itertools
import multiprocessing as mp
import os
import traceback
import yaml

from . import datamodel, mip, modelcreator, scenariogeneration, solutions, util

REQUIRED_FOR_PICKLE = scenariogeneration  # this prevents pycharm from removing this import, which is required for unpickling solutions


class AlibExecutionError(Exception):  pass


log = util.get_logger(__name__, make_file=False, propagate=True)

ALGORITHMS = {
    mip.ClassicMCFModel.ALGORITHM_ID: mip.ClassicMCFModel,
}


def register_algorithm(alg_id, alg_class):
    ALGORITHMS[alg_id] = alg_class


def run_experiment(experiment_yaml_file,
                   min_scenario_index, max_scenario_index,
                   concurrent):
    '''Entry point for running experiments.

    :param experiment_yaml_file: the yaml file detailing algorithm parameters / settings
    :param min_scenario_index:   the minimal scenario index that shall be included in the execution
    :param max_scenario_index:   the maximal scenario index that shall be included in the execution
    :param concurrent:           the number of processes (!= threads: each process may use multiple threads) to execute the experiments
    :return:
    '''
    log.info("PID: {}".format(os.getpid()))
    execution = ExperimentExecution(
        min_scenario_index, max_scenario_index,
        concurrent=concurrent
    )
    exp_data = yaml.load(experiment_yaml_file)
    scenario_picklefile = os.path.abspath(os.path.join(
        util.ExperimentPathHandler.INPUT_DIR, exp_data["SCENARIO_INPUT_PICKLE"])
    )
    with open(scenario_picklefile, "rb") as f:
        scenario_container = pickle.load(f)

    run_parameters = exp_data["RUN_PARAMETERS"]
    execution_parameter_container = ExecutionParameters(run_parameters)
    execution.setup(execution_parameter_container, scenario_container)

    result = execution.start_experiment()
    solution_storage_file = exp_data["RESULT_OUTPUT_PICKLE"]
    output_file = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR, solution_storage_file)
    log.info("Writing results to {}".format(output_file))
    with open(output_file, "wb") as f:
        pickle.dump(result, f)


class ExecutionParameters(object):
    '''Container to store execution parameters (and expand to multiple execution parameters)

    '''
    def __init__(self, execution_parameter_space):
        '''

        :param execution_parameter_space: dictionaries detailing parameters (from yaml file)
        '''
        self.execution_parameter_space = execution_parameter_space
        self.algorithm_parameter_list = []
        self.reverse_lookup = {}

    def generate_parameter_combinations(self):
        ''' Investigates the given dictionary structure and constructs the cartesian product of all parameters.

        :return:
        '''
        for algorithm in self.execution_parameter_space:
            alg_id = algorithm["ALGORITHM"]["ID"]
            self.reverse_lookup[alg_id] = {}
            # expand the parameter space of algorithm and gurobi parameters:
            alg_params = [{}]  # default to a single empty dictionary -> we need SOMETHING for the product below
            gurobi_params = [{}]
            if "ALGORITHM_PARAMETERS" in algorithm["ALGORITHM"]:
                alg_params = self._expand_innermost_parameter_space(
                    algorithm["ALGORITHM"]["ALGORITHM_PARAMETERS"]
                )
            if "GUROBI_PARAMETERS" in algorithm["ALGORITHM"]:
                gurobi_params = self._expand_innermost_parameter_space(
                    algorithm["ALGORITHM"]["GUROBI_PARAMETERS"]
                )

            parameter_list = [{"ALG_ID": alg_id, "GUROBI_PARAMETERS": p_grb, "ALGORITHM_PARAMETERS": p_alg}
                              for p_alg, p_grb in itertools.product(alg_params, gurobi_params)]
            self.algorithm_parameter_list.extend(parameter_list)

        for execution_id, parameters in enumerate(self.algorithm_parameter_list):
            alg_id = parameters["ALG_ID"]
            gurobi_params = parameters["GUROBI_PARAMETERS"]
            alg_params = parameters["ALGORITHM_PARAMETERS"]
            self.reverse_lookup.setdefault(alg_id, {})
            self.reverse_lookup[alg_id].setdefault("GUROBI_PARAMETERS", {})
            self.reverse_lookup[alg_id].setdefault("ALGORITHM_PARAMETERS", {})
            self.reverse_lookup[alg_id].setdefault("all", set())
            self.reverse_lookup[alg_id]["all"].add(execution_id)
            for key, value in gurobi_params.iteritems():
                self.reverse_lookup[alg_id]["GUROBI_PARAMETERS"].setdefault(key, {})
                self.reverse_lookup[alg_id]["GUROBI_PARAMETERS"][key].setdefault(value, set())
                self.reverse_lookup[alg_id]["GUROBI_PARAMETERS"][key][value].add(execution_id)
            for key, value in alg_params.iteritems():
                self.reverse_lookup[alg_id]["ALGORITHM_PARAMETERS"].setdefault(key, {})
                self.reverse_lookup[alg_id]["ALGORITHM_PARAMETERS"][key].setdefault(value, set())
                self.reverse_lookup[alg_id]["ALGORITHM_PARAMETERS"][key][value].add(execution_id)

    def _expand_innermost_parameter_space(self, parameter_space):
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

    def get_execution_ids(self, **kwargs):
        """ returns (suitable) execution ids filtered by **kwargs

        :param kwargs:
        :return:
        """
        if kwargs is not None:
            set_exec_ids = set()
            for i, (key, value) in enumerate(kwargs.iteritems()):
                set_exec_ids_single_lookup = set()
                if key == "ALG_ID":
                    set_exec_ids_single_lookup = self.reverse_lookup[value]["all"]
                elif key == "ALGORITHM_PARAMETERS":
                    for algo in self.reverse_lookup.keys():
                        for algo_para in value.keys():
                            if self.reverse_lookup[algo]["ALGORITHM_PARAMETERS"].has_key(algo_para):
                                result = self._helper_get_exec_id(self.reverse_lookup[algo]["ALGORITHM_PARAMETERS"][algo_para][value[algo_para]])
                                set_exec_ids_single_lookup = set_exec_ids_single_lookup.union(result)
                elif key == "GUROBI_PARAMETERS":
                    for algo in self.reverse_lookup.keys():
                        for gurobi_para in value.keys():
                            if self.reverse_lookup[algo]["GUROBI_PARAMETERS"].has_key(gurobi_para):
                                result = self._helper_get_exec_id(self.reverse_lookup[algo]["GUROBI_PARAMETERS"][gurobi_para][value[gurobi_para]])
                                set_exec_ids_single_lookup = set_exec_ids_single_lookup.union(result)
                if i == 0:
                    set_exec_ids = set_exec_ids_single_lookup
                else:
                    set_exec_ids = set_exec_ids.intersection(set_exec_ids_single_lookup)
            return set_exec_ids

    def _helper_get_exec_id(self, parameter_dict):
        results = set([])
        if type(parameter_dict) is set:
            return parameter_dict
        elif type(parameter_dict) is dict:
            for key, value in parameter_dict.iteritems():
                result = self._helper_get_exec_id(parameter_dict[key])
                results = results.union(result)
        return results


class ExperimentExecution(object):
    '''Handles the execution of experiments: slices scenario space and uses multiprocessing to obtain results.

    '''
    def __init__(self,
                 min_scenario_index,
                 max_scenario_index,
                 concurrent=1):

        self.scenario_container = None

        self.min_scenario_index = min_scenario_index
        self.max_scenario_index = max_scenario_index
        self.concurrent_executions = concurrent

        self.execution_parameters = None

        self.sss = None

        self.sss = solutions.ScenarioSolutionStorage(self.scenario_container, self.execution_parameters)
        self.pool = mp.Pool(self.concurrent_executions)

    def setup(self, execution_parameter_container, scenario_container):
        self.scenario_container = scenario_container
        self.execution_parameters = execution_parameter_container
        self.execution_parameters.generate_parameter_combinations()
        number_of_scenarios = len(self.scenario_container.scenario_list)
        if self.max_scenario_index > number_of_scenarios:
            log.warn("There are only {new} scenarios, restricting max_scenario_index parameter from {old} to {new}".format(
                old=self.max_scenario_index,
                new=number_of_scenarios
            ))
            self.max_scenario_index = self.min_scenario_index + number_of_scenarios
        #util.check_within_range(self.min_scenario_index, 0, self.max_scenario_index, none_allowed=False)
        util.check_int(self.min_scenario_index, False)
        util.check_int(self.max_scenario_index, False)

    def start_experiment(self):
        self.sss = solutions.ScenarioSolutionStorage(self.scenario_container, self.execution_parameters)
        for scenario_index in xrange(self.min_scenario_index, self.max_scenario_index):

            sp, scenario = self.scenario_container.scenario_triple[scenario_index]
            scenario.objective = datamodel.Objective.MAX_PROFIT
            self.sss.experiment_parameters = sp
            log.info("Scenario index {}  (Server Execution Range: {} -> {})".format(scenario_index,
                                                                                    self.min_scenario_index,
                                                                                    self.max_scenario_index))
            for execution_id, parameters in enumerate(self.execution_parameters.algorithm_parameter_list):
                args = (scenario_index, execution_id, parameters, scenario)
                self.pool.apply_async(_execute, args, callback=self._process_result)
                log.info("Submitted to task queue: Scenario {}, Alg {}, Execution {}:".format(
                    scenario_index, parameters["ALG_ID"], execution_id)
                )
                for key, param_dict in parameters.iteritems():
                    if key == "ALG_ID":
                        continue
                    log.debug("    {}:".format(key))
                    for param, values in param_dict.iteritems():
                        log.debug("        {} -> {}".format(param, values))
        self.pool.close()
        self.pool.join()
        return self.sss

    def _process_result(self, res):
        try:
            (scenario_id, execution_id, alg_result) = res
            log.info("Processing solution for {}, {}: {}".format(scenario_id, execution_id, alg_result))
            alg_id = self.execution_parameters.algorithm_parameter_list[execution_id]["ALG_ID"]
            with open("intermediate_result_{}_{}.pickle".format(scenario_id,alg_id), "wb") as f:
                pickle.dump((scenario_id,execution_id,alg_result), f)
            if alg_result is not None:
                # original_scenario = self.scenario_container.scenario_list[scenario_id]
                sp, original_scenario = self.scenario_container.scenario_triple[scenario_id]
                alg_result.cleanup_references(original_scenario)
                #while this might look a little bit weird, but we pickle the information again after the references have been cleaned up
                #as the function that actually cleans up the references might fail...
                with open("intermediate_result_{}_{}.pickle".format(scenario_id, alg_id), "wb") as f:
                    pickle.dump((scenario_id, execution_id, alg_result), f)
            self.sss.add_solution(alg_id, scenario_id, execution_id, alg_result)
        except Exception as e:
            stacktrace = ("\nError in processing algorithm result {}:\n".format(res) +
                          traceback.format_exc(limit=100))
            for line in stacktrace.split("\n"):
                log.error(line)
            raise e


def _initialize_algorithm(scenario, logger, parameters):
    alg_class = ALGORITHMS[parameters["ALG_ID"]]
    gurobi_settings = None
    if parameters["GUROBI_PARAMETERS"]:
        gurobi_settings = modelcreator.GurobiSettings(**parameters["GUROBI_PARAMETERS"])
    alg_instance = alg_class(scenario, logger=logger, gurobi_settings=gurobi_settings, **parameters["ALGORITHM_PARAMETERS"])
    return alg_instance


def _execute(scenario_id, execution_id, parameters, scenario):
    """
    This function is submitted to the processing pool

    :param scenario_id:
    :param execution_id:
    :param algorithm_instance:
    :return:
    """
    logger = util.get_logger("worker_{}".format(os.getpid()), propagate=False)

    try:
        algorithm_instance = _initialize_algorithm(scenario, logger, parameters)
        logger.info("Processing: Algorithm {}, Scenario {}, Execution {}".format(
            algorithm_instance.__class__.ALGORITHM_ID, scenario_id, execution_id
        ))
        algorithm_instance.init_model_creator()
        alg_solution = algorithm_instance.compute_integral_solution()

        logger.info("Finished computation. Result:".format(mp.current_process().name))
        if alg_solution is not None:
            for line in str(alg_solution.get_solution()).split("\n"):
                logger.debug("    " + line)
        else:
            logger.debug("No feasible solution was found!")

        if hasattr(algorithm_instance, "model"):
            del algorithm_instance.model
        if hasattr(algorithm_instance, "mc"):
            if hasattr(algorithm_instance.mc, "model"):
                del algorithm_instance.mc.model

        del algorithm_instance

        execution_result = (scenario_id, execution_id, alg_solution)

        return execution_result
    except Exception as e:
        stacktrace = ("\nError in scenario {}, execution {}:\n".format(scenario_id, execution_id) +
                      traceback.format_exc(limit=100))
        # print stacktrace
        for line in stacktrace.split("\n"):
            logger.error(line)
        raise e
