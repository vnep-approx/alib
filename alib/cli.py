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

import os
import yaml
import click
import itertools
import logging

try:
    import cPickle as pickle
except ImportError:
    import pickle

from . import run_experiment, scenariogeneration, solutions, util, datamodel

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions


@click.group()
def cli():
    pass

import logging
def initialize_logger(filename, log_level_print, log_level_file, allow_override=False):
    log_level_print = logging._levelNames[log_level_print.upper()]
    log_level_file = logging._levelNames[log_level_file.upper()]
    util.initialize_root_logger(filename, log_level_print, log_level_file, allow_override=allow_override)


@cli.command()
@click.argument('scenario_output_file')
@click.argument('parameters', type=click.File('r'))
@click.option('--threads', default=1)
@click.option('--scenario_index_offset', default=0)
def generate_scenarios(scenario_output_file, parameters, threads, scenario_index_offset):
    f_generate_scenarios(scenario_output_file, parameters, threads, scenario_index_offset)


def f_generate_scenarios(scenario_output_file, parameter_file, threads, scenario_index_offset=0):
    """
    Generates the scenarios according to the scenario parameters found in the parameter_file.

    This function is separated from generate_scenarios so that it can be reused when extending the CLI from outside
    the alib.

    :param scenario_output_file: path to pickle file to which the resulting scenarios will be written
    :param parameter_file: readable file object containing the scenario parameters in yml format
    :param threads: number of concurrent threads used for scenario generation
    :param scenario_index_offset: offset that is added to every scenario ID. Useful when extending existing scenario sets.
    :return:
    """
    click.echo('Generate Scenarios')
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(parameter_file.name).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_scenario_generation.log".format(file_basename))
    util.initialize_root_logger(log_file)
    scenariogeneration.generate_pickle_from_yml(parameter_file, scenario_output_file, threads, scenario_index_offset=scenario_index_offset)


@cli.command()
@click.argument('pickle_file', type=click.File('r'))
@click.option('--col_output_limit', default=None)
def pretty_print(pickle_file, col_output_limit):
    data = pickle.load(pickle_file)
    pp = util.PrettyPrinter()
    print pp.pprint(data, col_output_limit=col_output_limit)


@cli.command()
@click.argument('experiment_yaml', type=click.File('r'))
@click.argument('min_scenario_index', type=click.INT)
@click.argument('max_scenario_index', type=click.INT)
@click.option('--concurrent', default=1, help="number of processes to be used in parallel")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
@click.option('--shuffle_instances/--original_order', is_flag=True, default=True, help="shall instances be shuffled or ordered according to their ids (ascendingly)")
@click.option('--overwrite_existing_temporary_scenarios/--use_existing_temporary_scenarios', is_flag=True, default=False, help="shall existing temporary scenario files be overwritten or used?")
@click.option('--overwrite_existing_intermediate_solutions/--use_existing_intermediate_solutions', is_flag=True, default=False, help="shall existing intermediate solution files be overwritten or used?")
@click.option('--remove_temporary_scenarios/--keep_temporary_scenarios', is_flag=True, default=False, help="shall temporary scenario files be removed after execution?")
@click.option('--remove_intermediate_solutions/--keep_intermediate_solutions', is_flag=True, default=False, help="shall intermediate solutions be removed after execution?")
def start_experiment(experiment_yaml,
                     min_scenario_index,
                     max_scenario_index,
                     concurrent,
                     log_level_print,
                     log_level_file,
                     shuffle_instances,
                     overwrite_existing_temporary_scenarios,
                     overwrite_existing_intermediate_solutions,
                     remove_temporary_scenarios,
                     remove_intermediate_solutions
                     ):
    f_start_experiment(experiment_yaml,
                       min_scenario_index,
                       max_scenario_index,
                       concurrent,
                       log_level_print,
                       log_level_file,
                       shuffle_instances,
                       overwrite_existing_temporary_scenarios,
                       overwrite_existing_intermediate_solutions,
                       remove_temporary_scenarios,
                       remove_intermediate_solutions
                       )


def f_start_experiment(experiment_yaml,
                       min_scenario_index,
                       max_scenario_index,
                       concurrent,
                       log_level_print,
                       log_level_file,
                       shuffle_instances=True,
                       overwrite_existing_temporary_scenarios=False,
                       overwrite_existing_intermediate_solutions=False,
                       remove_temporary_scenarios=False,
                       remove_intermediate_solutions=False
                       ):
    """
    Executes the experiment according to the execution parameters found in the experiment_yaml.

    This function is separated from start_experiment so that it can be reused when extending the CLI from outside
    the alib.

    :param experiment_yaml: readable file object containing the execution parameters in yml format
    :param scenario_output_file: path to pickle file to which the resulting scenarios will be written
    :param threads: number of concurrent threads used for scenario generation
    :return:
    """
    click.echo('Start Experiment')
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(experiment_yaml.name).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_experiment_execution.log".format(file_basename))

    initialize_logger(log_file, log_level_print, log_level_file)

    run_experiment.run_experiment(
        experiment_yaml,
        min_scenario_index, max_scenario_index,
        concurrent,
        shuffle_instances,
        overwrite_existing_temporary_scenarios,
        overwrite_existing_intermediate_solutions,
        remove_temporary_scenarios,
        remove_intermediate_solutions
    )


@cli.command()
@click.argument('yaml_file_with_cacus_request_graph_definition', type=click.Path())
@click.option('--iterations', type=click.INT, default=100000)
def inspect_cactus_request_graph_generation(yaml_file_with_cacus_request_graph_definition,
                                            iterations):
    util.ExperimentPathHandler.initialize()
    print(yaml_file_with_cacus_request_graph_definition)
    param_space = None
    with open(yaml_file_with_cacus_request_graph_definition, "r") as f:
        param_space = yaml.load(f)
    print "----------------------"
    print param_space
    print "----------------------"
    for request_generation_task in param_space[scenariogeneration.REQUEST_GENERATION_TASK]:
        for name, values in request_generation_task.iteritems():
            print name, values
            if "CactusRequestGenerator" in values:
                raw_parameters = values["CactusRequestGenerator"]
                print "\n\nextracted the following parameters..."
                print name, ": ", raw_parameters
                f_inspect_specfic_cactus_request_graph_generation_and_output(name, raw_parameters, iterations)


def f_inspect_specfic_cactus_request_graph_generation_and_output(name, raw_parameters, iterations):
    simple_substrate = datamodel.Substrate("stupid_simple")
    simple_substrate.add_node("u", ["universal"], capacity={"universal": 1000}, cost=1000)
    simple_substrate.add_node("v", ["universal"], capacity={"universal": 1000}, cost=1000)
    simple_substrate.add_edge("u", "v", capacity=1000, cost=1000, bidirected=True)

    # flatten values
    param_key_list = []
    param_value_list = []
    for key, value in raw_parameters.iteritems():
        param_key_list.append(key)
        # only the following parameters really define the graphs generated.
        # hence for these the original lists are preserved, while for the other parameters
        # only the last parameter is kept (hoping that it has the `largest` value)
        if key in ["branching_distribution",
                   "min_number_of_nodes",
                   "max_number_of_nodes",
                   "layers",
                   "probability",
                   "max_cycles",
                   "iterations"]:
            param_value_list.append(value)
        else:
            param_value_list.append([value[-1]])

    import numpy as np
    import matplotlib.pyplot as plt

    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        return xs, ys

    for param_combo_index, param_combo in enumerate(itertools.product(*param_value_list)):
        flattended_raw_parameters = {}
        for index, value in enumerate(param_combo):
            flattended_raw_parameters[param_key_list[index]] = value

        print flattended_raw_parameters
        cactus_generator = scenariogeneration.CactusRequestGenerator()

        advanced_information = cactus_generator.advanced_empirical_number_of_nodes_edges(flattended_raw_parameters, simple_substrate, iterations)

        min_nodes = flattended_raw_parameters["min_number_of_nodes"]
        max_nodes = flattended_raw_parameters["max_number_of_nodes"]

        edge_count_per_node = {node_number: [] for node_number in range(min_nodes, max_nodes + 1)}

        node_numbers = []
        edge_numbers = []

        max_edge_count = 0
        for node, edge in advanced_information.node_edge_comination:
            edge_count_per_node[node].append(edge)
            if edge > max_edge_count:
                max_edge_count = edge
            node_numbers.append(node)
            edge_numbers.append(edge)

        fig = plt.figure(figsize=(8, 12))
        ax = plt.subplot(111)

        for node_number in range(min_nodes, max_nodes + 4):
            if node_number == max_nodes + 1:
                xs, ys = ecdf(node_numbers)
                xs = np.insert(xs, 0, min_nodes, axis=0)
                xs = np.insert(xs, 0, min_nodes - 1, axis=0)
            elif node_number == max_nodes + 2:
                xs, ys = ecdf(edge_numbers)
                xs = np.insert(xs, 0, min_nodes, axis=0)
                xs = np.insert(xs, 0, min_nodes - 1, axis=0)
            elif node_number == max_nodes + 3:
                xs, ys = ecdf(advanced_information.generated_cycles)
                xs = np.insert(xs, 0, 1, axis=0)
                xs = np.insert(xs, 0, 0, axis=0)
            else:
                if len(edge_count_per_node[node_number]) == 0:
                    continue
                xs, ys = ecdf(edge_count_per_node[node_number])
                xs = np.insert(xs, 0, min(edge_count_per_node[node_number]), axis=0)
                xs = np.insert(xs, 0, min(edge_count_per_node[node_number]) - 1, axis=0)

            xs = np.append(xs, max_edge_count + 1)
            ys = np.insert(ys, 0, 0, axis=0)
            ys = np.insert(ys, 0, 0, axis=0)
            ys = np.append(ys, 1.0)
            # print node_number, xs, ys
            # print xs, ys
            print "plot ....", node_number
            label = "edge_count_per_node_count_{}".format(node_number)
            if node_number == max_nodes + 1:
                label = "node_count"
            if node_number == max_nodes + 2:
                label = "edge_count"
                print xs[0:10], ys[0:10]
            if node_number == max_nodes + 3:
                label = "cycle_count"
            if node_number <= max_nodes:
                ax.step(xs, ys, label=label, linestyle="-.")
            else:
                ax.step(xs, ys, label=label, linestyle="-")

        title = "\n".join(["{}: {}".format(key, value) for key, value in flattended_raw_parameters.iteritems()])
        title += "\n\nExp. |V|: {}; Exp. |E|: {}; Exp. |C|: {}; Exp. CC: {}\n\n".format(
            advanced_information.nodes_generated / float(iterations),
            advanced_information.edges_generated / float(iterations),
            sum(advanced_information.generated_cycles) / float(iterations),
            advanced_information.overall_cycle_edges / float(advanced_information.edges_generated))
        title += "failed generation attempts: {}%\n".format(advanced_information.generation_tries_failed / float(advanced_information.generation_tries_overall) * 100.0)
        title += "overall iterations: {}".format(iterations)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=2)
        plt.title(title)
        plt.xticks(range(0, max_edge_count + 1))
        plt.tight_layout()
        # plt.show()
        filename = util.ExperimentPathHandler.OUTPUT_DIR + "/{}_{}.pdf".format(name, param_combo_index)
        print filename
        plt.savefig(filename, dpi=300)

        # overall_cycle_edges /= float(total_edges)
        # total_nodes /= float(iterations)
        # total_edges /= float(iterations)
        #
        # print("Expecting {} nodes, {} edges, {}% edges on cycle".format(total_nodes, total_edges,
        #                                                                 overall_cycle_edges * 100))

        # print edge_count_per_node


@cli.command()
@click.argument('sss_pickle_file_1', type=click.Path(exists=True, dir_okay=False))
@click.argument('sss_pickle_file_2', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
def merge_sss(sss_pickle_file_1, sss_pickle_file_2, output):
    f_merge_sss(sss_pickle_file_1, sss_pickle_file_2, output)


def f_merge_sss(sss_pickle_file_1, sss_pickle_file_2, output):
    with open(sss_pickle_file_1, "rb") as f:
        sss_1 = pickle.load(f)
    with open(sss_pickle_file_2, "rb") as f:
        sss_2 = pickle.load(f)
    if (not isinstance(sss_1, solutions.ScenarioSolutionStorage)
            or not isinstance(sss_2, solutions.ScenarioSolutionStorage)):
        raise ValueError("Expected pickle files for two ScenarioSolutionStorage instances!")

    sss_1.merge_with_other_sss(sss_2)

    with open(output, "wb") as f:
        pickle.dump(sss_1, f)


@cli.command()
@click.argument('scenario_container_pickle_file_1', type=click.Path(exists=True, dir_okay=False))
@click.argument('scenario_container_pickle_file_2', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
def merge_scenario_containers(scenario_container_pickle_file_1, scenario_container_pickle_file_2, output):
    f_merge_scenario_containers(scenario_container_pickle_file_1, scenario_container_pickle_file_2, output)


def f_merge_scenario_containers(scenario_container_pickle_file_1, scenario_container_pickle_file_2, output):
    with open(scenario_container_pickle_file_1, "rb") as f:
        scenario_container_1 = pickle.load(f)
    with open(scenario_container_pickle_file_2, "rb") as f:
        scenario_container_2 = pickle.load(f)
    if (not isinstance(scenario_container_1, scenariogeneration.ScenarioParameterContainer)
            or not isinstance(scenario_container_2, scenariogeneration.ScenarioParameterContainer)):
        raise ValueError("Expected pickle files for two ScenarioParameterContainer instances!")

    scenario_container_1.merge_with_other_scenario_parameter_container(scenario_container_2)

    with open(output, "wb") as f:
        pickle.dump(scenario_container_1, f)


@cli.command()
@click.option('--min_number_nodes', type=click.INT, default=10)
@click.option('--max_number_nodes', type=click.INT, default=100000)
def summarize_topology_zoo_graphs(min_number_nodes, max_number_nodes):
    scenariogeneration.summarize_topology_zoo_graphs(min_number_nodes, max_number_nodes)


@cli.command()
@click.argument('gml_path', type=click.Path())
@click.argument('yml_path', type=click.Path())
@click.option('--consider_disconnected/--discard_disconnected', default="True")
def convert_topology_zoo_gml_to_yml(gml_path, yml_path, consider_disconnected):
    scenariogeneration.convert_topology_zoo_gml_to_yml(gml_path, yml_path, consider_disconnected)

if __name__ == '__main__':
    cli()
