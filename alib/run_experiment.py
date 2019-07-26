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
from collections import deque
import itertools
import multiprocessing as mp
import Queue
import os
import traceback
import yaml
import random
from datetime import datetime

from . import datamodel, mip, modelcreator, scenariogeneration, solutions, util

REQUIRED_FOR_PICKLE = scenariogeneration  # this prevents pycharm from removing this import, which is required for unpickling solutions


class AlibExecutionError(Exception):  pass


log = util.get_logger(__name__, make_file=False, propagate=True)

ALGORITHMS = {
    mip.ClassicMCFModel.ALGORITHM_ID: mip.ClassicMCFModel,
}


def register_algorithm(alg_id, alg_class):
    ALGORITHMS[alg_id] = alg_class

class CustomizedDataManager(mp.managers.SyncManager):
    pass

CustomizedDataManager.register('ScenarioParameterContainer', scenariogeneration.ScenarioParameterContainer)
CustomizedDataManager.register('ScenarioSolutionStorage', solutions.ScenarioSolutionStorage)



def run_experiment(experiment_yaml_file,
                   min_scenario_index,
                   max_scenario_index,
                   concurrent,
                   shuffle_instances=True,
                   overwrite_existing_temporary_scenarios=False,
                   overwrite_existing_intermediate_solutions=False,
                   remove_temporary_scenarios=False,
                   remove_intermediate_solutions=False
                   ):
    '''Entry point for running experiments.

    :param experiment_yaml_file: the yaml file detailing algorithm parameters / settings
    :param min_scenario_index:   the minimal scenario index that shall be included in the execution
    :param max_scenario_index:   the maximal scenario index that shall be included in the execution
    :param concurrent:           the number of processes (!= threads: each process may use multiple threads) to execute the experiments
    :param shuffle_instances:    shall the instances be shuffled (deterministically) to better mix of hard and simple instances
    :param overwrite_existing_intermediate_scenario_pickles:    shall existing scenario pickle files be replaced?
    :param read_existing_intermediate_solutions_from_file:      shall existing intermediate solution files be used or shall instance solutions be recomputed?
    :param remove_temporary_scenarios:          shall temporary scenario files be removed after execution?
    :param remove_intermediate_solutions:       shall intermediate solution files be removed after execution?

    :return:
    '''
    log.info("PID: {}".format(os.getpid()))
    execution = ExperimentExecution(
        min_scenario_index,
        max_scenario_index,
        concurrent=concurrent,
        shuffle_instances=shuffle_instances,
        overwrite_existing_temporary_scenarios=overwrite_existing_temporary_scenarios,
        overwrite_existing_intermediate_solutions=overwrite_existing_intermediate_solutions,
        remove_temporary_scenarios=remove_temporary_scenarios,
        remove_intermediate_solutions=remove_intermediate_solutions
    )
    exp_data = yaml.load(experiment_yaml_file)
    scenario_picklefile = os.path.abspath(os.path.join(
        util.ExperimentPathHandler.INPUT_DIR, exp_data["SCENARIO_INPUT_PICKLE"])
    )

    run_parameters = exp_data["RUN_PARAMETERS"]
    execution_parameter_container = ExecutionParameters(run_parameters)
    execution.setup(execution_parameter_container, scenario_picklefile)

    result = execution.start_experiment()
    solution_storage_file = exp_data["RESULT_OUTPUT_PICKLE"]
    output_file = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR, solution_storage_file)
    log.info("Writing results to {}".format(output_file))
    with open(output_file, "wb") as f:
        pickle.dump(result, f)
    execution.clean_up()




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
        :param parameter_space: dictionary
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
                 concurrent=1,
                 shuffle_instances=True,
                 overwrite_existing_temporary_scenarios=False,
                 overwrite_existing_intermediate_solutions=False,
                 remove_temporary_scenarios=False,
                 remove_intermediate_solutions=False
                 ):

        self.min_scenario_index = min_scenario_index
        self.max_scenario_index = max_scenario_index
        self.concurrent_executions = concurrent
        self.shuffle_instances = shuffle_instances
        self.overwrite_existing_temporary_scenarios = overwrite_existing_temporary_scenarios
        self.overwrite_existing_intermediate_solutions = overwrite_existing_intermediate_solutions
        self.remove_temporary_scenarios = remove_temporary_scenarios
        self.remove_intermediate_solutions = remove_intermediate_solutions

        self.execution_parameters = None

        self.created_temporary_scenario_files = []
        self.created_intermediate_solution_files = []


        self.process_indices = [i for i in range(concurrent)]
        self.processes = {i : None for i in self.process_indices}
        self.process_args = {i : None for i in self.process_indices}
        self.input_queues = {i : mp.Queue() for i in self.process_indices}
        self.result_queue = mp.Queue()
        self.error_queue = mp.Queue()
        self.unprocessed_tasks = deque()
        self.finished_tasks = deque()
        self.currently_active_processes = 0
        self.current_scenario = {i : None for i in self.process_indices}

        self.sss = None

    def setup(self, execution_parameter_container, scenario_picklefile):
        self.scenario_picklefile = scenario_picklefile
        self.execution_parameters = execution_parameter_container
        self.execution_parameters.generate_parameter_combinations()

        scenario_container = self._load_scenario_container()

        number_of_scenarios = len(scenario_container.scenario_list)
        if self.max_scenario_index > number_of_scenarios:
            log.warn("There are only {new} scenarios, restricting max_scenario_index parameter from {old} to {new}".format(
                old=self.max_scenario_index,
                new=number_of_scenarios
            ))
            self.max_scenario_index = self.min_scenario_index + number_of_scenarios
        #util.check_within_range(self.min_scenario_index, 0, self.max_scenario_index, none_allowed=False)
        util.check_int(self.min_scenario_index, False)
        util.check_int(self.max_scenario_index, False)

        for scenario_index in xrange(self.min_scenario_index, self.max_scenario_index):

            sp, scenario = scenario_container.scenario_triple[scenario_index]

            # According to comments at scenariogeneration.build_scenario, if a profit calculation task is missing, the objective should
            # be minimization. This might not be needed at all.
            if scenariogeneration.PROFIT_CALCULATION_TASK in sp:
                scenario.objective = datamodel.Objective.MAX_PROFIT

            log.info("Scenario index {}  (Server Execution Range: {} -> {})".format(scenario_index,
                                                                                    self.min_scenario_index,
                                                                                    self.max_scenario_index))

            scenario_filename = self._get_scenario_pickle_filename(scenario_index)
            if not os.path.exists(scenario_filename) or not self.overwrite_existing_temporary_scenarios:
                with open(scenario_filename, "w") as f:
                    pickle.dump(scenario, f)
            self.created_temporary_scenario_files.append(scenario_filename)

            for execution_id, parameters in enumerate(self.execution_parameters.algorithm_parameter_list):

                args = (scenario_index, execution_id, parameters)

                intermediate_solution_filename = self._get_scenario_solution_filename(scenario_index, execution_id)

                if self.overwrite_existing_intermediate_solutions or not os.path.exists(intermediate_solution_filename):
                    self.unprocessed_tasks.append(args)
                    self.created_intermediate_solution_files.append(intermediate_solution_filename)

                    log.info("Stored unprocessed task into list: Scenario {}, Alg {}, Execution {}:".format(
                        scenario_index, parameters["ALG_ID"], execution_id))
                else:
                    log.info("Will not execute the following, as an intermediate solution file already exists.\n"
                             "Scenario {}, Alg {}, Execution {}".format(scenario_index, parameters["ALG_ID"], execution_id))
                    self.finished_tasks.append((scenario_index, execution_id, None))

                for key, param_dict in parameters.iteritems():
                    if key == "ALG_ID":
                        continue
                    log.debug("    {}:".format(key))
                    for param, values in param_dict.iteritems():
                        log.debug("        {} -> {}".format(param, values))

        if self.shuffle_instances:
            random.seed(0)
            random.shuffle(self.unprocessed_tasks)

        del scenario_container


    def start_experiment(self):

        self._spawn_processes()

        self._retrieve_results_and_spawn_processes()

        self._collect_results()

        return self.sss

    def _spawn_processes(self):
        while self.currently_active_processes < self.concurrent_executions and len(self.unprocessed_tasks) > 0:
            args = self.unprocessed_tasks.popleft()
            self._spawn_process(args)

    def _spawn_process(self, args):
        scenario_index, execution_id, parameters = args

        scenario = self._load_scenario(scenario_index)

        for process_id, process in self.processes.iteritems():
            if process is None:
                extended_args = scenario_index, execution_id, parameters, scenario, process_id, self.result_queue, self.error_queue
                self.processes[process_id] = mp.Process(target=_execute, args=extended_args)
                log.info("Spawning process with index {}".format(process_id))
                self.processes[process_id].start()
                self.currently_active_processes += 1
                self.current_scenario[process_id] = scenario
                break
            # else:
            #     log.info("Process with index {} is currently still active".format(process_id))

    def _handle_finished_process(self, scenario_id, execution_id, process_index, failed):
        self.finished_tasks.append((scenario_id, execution_id, failed))
        # joining without timeout might cause the whole execution to wait for the slowest process until the errors/results of other
        # processes are handled or new ones created.
        self.processes[process_index].join()
        # terminate might corrupt the queues or interrupt exception handling...
        # self.processes[process_index].terminate()
        self.processes[process_index] = None
        self.currently_active_processes -= 1
        # set the current scenario to invalid, it is used when processing the result, which should be done before calling this function
        self.current_scenario[process_index] = None

    def _retrieve_results_and_spawn_processes(self):
        while self.currently_active_processes > 0:
            try:
                excetion_info = self.error_queue.get_nowait()
                scenario_id, execution_id, exception, process_index = excetion_info
                # NOTE: assertion error happens often when memory allocation failed for the treewidth calculation
                log.error("Exception {} at process {} of scenario {}, execution id {}. Skipping executing scenario!".
                          format(exception, process_index, scenario_id, execution_id))
                # we should not expect solution from this process
                self._handle_finished_process(scenario_id, execution_id, process_index, failed=True)
            except Queue.Empty as e:
                log.debug("No error found in error queue.")

            try:
                result = self.result_queue.get(timeout=10)

                scenario_id, execution_id, alg_result, process_index = result

                self._process_result(result)

                self._handle_finished_process(scenario_id, execution_id, process_index, failed=False)
            except Queue.Empty as e:
                log.debug("No result found in result queue yet, retrying in 10s... "
                          "Current processes: {}".format(self.processes))

            for process_index, process in self.processes.items():
                if process is not None:
                    if process.exitcode is not None:
                        # processes with handled exception also has exit code 0
                        if process.exitcode < 0:
                            # only those have such which were terminated by some external source, without being able to finish.
                            log.warn("Discarding terminated inactive process with process id {}: {}".format(process_index, process))
                            self.processes[process_index] = None
                            self.currently_active_processes -= 1
                            self.current_scenario[process_index] = None

            self._spawn_processes()

    def _process_result(self, res):
        try:
            scenario_id, execution_id, alg_result, process_index = res
            log.info("Processing solution for scenario {}, execution id {}, result: {}".format(scenario_id, execution_id, alg_result))

            self._dump_scenario_solution(scenario_id, execution_id, (scenario_id, execution_id, alg_result))

            if alg_result is not None:
                # original_scenario = self.scenario_container.scenario_list[scenario_id]
                original_scenario = self.current_scenario[process_index]
                alg_result.cleanup_references(original_scenario)
                # while this might look a little bit weird, but we pickle the information again after the references have been cleaned up
                # as the function that actually cleans up the references might fail...
                self._dump_scenario_solution(scenario_id, execution_id, (scenario_id, execution_id, alg_result))

        except Exception as e:
            # if error occurs in result processing, we want it to break the execution
            stacktrace = ("\nError in processing algorithm result {}:\n".format(res) +
                          traceback.format_exc(limit=100))
            for line in stacktrace.split("\n"):
                log.error(line)
            raise e

    def _collect_results(self):
        scenario_container = self._load_scenario_container()
        self.sss = solutions.ScenarioSolutionStorage(scenario_container, self.execution_parameters)

        for finished_scenario_id, finished_execution_id, is_failed in self.finished_tasks:
            if is_failed is None or is_failed is False:
                alg_id = self.execution_parameters.algorithm_parameter_list[finished_execution_id]["ALG_ID"]
                intermediate_solution_filename = self._get_scenario_solution_filename(finished_scenario_id, finished_execution_id)

                log.info("Collecting result stored in file {}".format(intermediate_solution_filename))
                scenario_solution = self._load_scenario_solution(finished_scenario_id, finished_execution_id)
                sp, scenario = scenario_container.scenario_triple[finished_scenario_id]
                self.sss.experiment_parameters = sp

                scenario_id, execution_id, alg_result = self._load_scenario_solution(finished_scenario_id, finished_execution_id)
                # FIXME: if a result was not found we have nothing to do here
                if alg_result is not None:
                    # IMPORTANT:    this cleanup is necessary as after loading the pickle the original scenario does not match
                    #               the pickled one!
                    alg_result.cleanup_references(scenario)


                    self.sss.add_solution(alg_id, scenario_id, execution_id, alg_result)
                else:
                    log.info("Skipping scenario {} with execution id {} reference cleanup and solution addition, because no solution was "
                             "found.".format(finished_scenario_id, finished_execution_id))
            elif is_failed is True:
                log.info("Skipping reading non existing solution file of failed scenario {}, execution id {}".format(
                    finished_scenario_id, finished_execution_id))
            else:
                raise Exception("Dunno what is happening here!")


    def clean_up(self):
        log.info("Cleaning up..")
        #remove created temporary scenario files and intermediate solution files if these were created
        if self.remove_temporary_scenarios:
            for temp_scenario_file in self.created_temporary_scenario_files:
                if os.path.exists(temp_scenario_file):
                    log.info("Removing {}..".format(temp_scenario_file))
                    os.remove(temp_scenario_file)
                else:
                    log.warning("Wanted to remove {}, but file didn't exist".format(temp_scenario_file))
        if self.remove_intermediate_solutions:
            for intermediate_solution_file in self.created_intermediate_solution_files:
                if os.path.exists(intermediate_solution_file):
                    log.info("Removing {}..".format(intermediate_solution_file))
                    os.remove(intermediate_solution_file)
                else:
                    log.warning("Wanted to remove {}, but file didn't exist".format(intermediate_solution_file))
        log.info("\tdone.")


    def _load_scenario_container(self):
        scenario_container = None
        with open(self.scenario_picklefile, "r") as f:
            scenario_container = pickle.load(f)
        return scenario_container


    def _get_scenario_pickle_filename(self, scenario_id):
        return "temp_scenario_{}.pickle".format(scenario_id)

    def _load_scenario(self, scenario_id):
        scenario = None
        with open(self._get_scenario_pickle_filename(scenario_id), "rb") as f:
            scenario = pickle.load(f)
        return scenario

    def _dump_scenario(self, scenario_id, scenario):
        with open(self._get_scenario_pickle_filename(scenario_id), "wb") as f:
            pickle.dump(scenario, f)

    def _get_scenario_solution_filename(self, scenario_id, execution_id):
        return "intermediate_result_{}_{}.pickle".format(scenario_id, execution_id)

    def _load_scenario_solution(self, scenario_id, execution_id):
        scenario = None
        with open(self._get_scenario_solution_filename(scenario_id, execution_id), "rb") as f:
            scenario = pickle.load(f)
        return scenario

    def _dump_scenario_solution(self, scenario_id, execution_id, scenario_solution):
        scenario = None
        with open(self._get_scenario_solution_filename(scenario_id, execution_id), "wb") as f:
            pickle.dump(scenario_solution, f)


def _initialize_algorithm(scenario, logger, parameters):
    alg_class = ALGORITHMS[parameters["ALG_ID"]]
    gurobi_settings = None
    if parameters["GUROBI_PARAMETERS"]:
        gurobi_settings = modelcreator.GurobiSettings(**parameters["GUROBI_PARAMETERS"])
    alg_instance = alg_class(scenario, logger=logger, gurobi_settings=gurobi_settings, **parameters["ALGORITHM_PARAMETERS"])
    return alg_instance


def _execute(scenario_id, execution_id, parameters, scenario, process_index, result_queue, error_queue):
    """
    This function is submitted to the processing pool

    :param scenario_id:
    :param execution_id:
    :param algorithm_instance:
    :return:
    """

    logger_filename = "worker_{}".format(os.getpid())

    logger = util.get_logger(logger_filename, propagate=False)

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

        execution_result = (scenario_id, execution_id, alg_solution, process_index)

        result_queue.put(execution_result)

    except Exception as e:
        stacktrace = ("\nError in scenario {}, execution {}:\n".format(scenario_id, execution_id) +
                      traceback.format_exc(limit=100))
        # print stacktrace
        for line in stacktrace.split("\n"):
            logger.error(line)
        exception_info = (scenario_id, execution_id, str(type(e)), process_index)
        # instead of raising the exception save it to the parent process
        error_queue.put(exception_info)
    finally:
        # in any case we need to move the logger file to finished
        logger_filename_orig = util.get_logger_filename(logger_filename)
        current_time = datetime.now()

        logger_filename_finished = util.get_logger_filename("finished" + current_time.strftime("_%Y_%m_%d_%H_%M_%S_") + logger_filename)

        os.rename(logger_filename_orig, logger_filename_finished)
