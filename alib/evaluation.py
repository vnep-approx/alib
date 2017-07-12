"""This is the evaluation and plotting module.

This module handles all plotting related evaluation.
"""

import os
import pickle
import sys
from collections import namedtuple
from itertools import combinations, product
from time import gmtime, strftime

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

from . import solutions

sys.path.insert(0, os.path.abspath('../'))

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions

ReducedSolutionSigmetrics = namedtuple("ReducedSolutionSigmetrics",
                                       "load runtime status found_solution embedding_ratio temporal_log nu_real_req original_number_requests")


def reduce_solution(solution_pickle):
    with open(solution_pickle, "rb") as input_file:
        solution = pickle.load(input_file)
    file_basename = os.path.basename(input_file.name).split(".")[0].lower()
    ssd = solution.algorithm_scenario_solution_dictionary
    for algorithm in ssd.keys():
        for scenario_id in ssd[algorithm].keys():
            for exec_id in ssd[algorithm][scenario_id].keys():
                scenario = solution.scenario_parameter_container.scenario_list[scenario_id]
                load = dict([((u, v), 0.0) for (u, v) in scenario.substrate.edges])
                for u in scenario.substrate.nodes:
                    for types in scenario.substrate.node[u]['supported_types']:
                        load[(types, u)] = 0.0
                mappings = ssd[algorithm][scenario_id][exec_id].solution.request_mapping
                number_of_embedde_reqs = 0
                number_of_req_profit = 0
                number_of_requests = len(ssd[algorithm][scenario_id][exec_id].solution.scenario.requests)
                for req in ssd[algorithm][scenario_id][exec_id].solution.scenario.requests:
                    if req.profit > 0.001:
                        number_of_req_profit += 1
                    if mappings[req].is_embedded:
                        number_of_embedde_reqs += 1
                        for i, u in mappings[req].mapping_nodes.iteritems():
                            # print "req node", i
                            # print "sub node", u
                            # print "type of i" ,req.get_type(i)
                            # print "demand", req.get_node_demand(i)
                            node_demand = req.get_node_demand(i)
                            load[(req.get_type(i), u)] += node_demand
                        for ve, sedge_list in mappings[req].mapping_edges.iteritems():
                            # print "req_edge", ve
                            # print "sub edge list", sedge_list
                            # print "demand of edge", req.get_edge_demand(ve)
                            edge_demand = req.get_edge_demand(ve)
                            for sedge in sedge_list:
                                load[sedge] = edge_demand
                percentage_embbed = number_of_embedde_reqs / float(number_of_requests)
                algo_result = ssd[algorithm][scenario_id][exec_id]
                ssd[algorithm][scenario_id][exec_id] = ReducedSolutionSigmetrics(
                    load=load,
                    runtime=algo_result.temporal_log.log_entries[-1].time_within_gurobi,
                    status=algo_result.status,
                    found_solution=None,
                    embedding_ratio=percentage_embbed,
                    temporal_log=algo_result.temporal_log,
                    nu_real_req=number_of_req_profit,
                    original_number_requests=number_of_requests
                )
    del solution.scenario_parameter_container.scenario_list
    del solution.scenario_parameter_container.scenario_triple
    with open(file_basename + "_reduced.pickle", "wb") as f:
        pickle.dump(solution, f)


def load_reduced_pickle(reduced_pickle):
    with open(reduced_pickle, "rb") as f:
        data = pickle.load(f)
    return data


def get_filename(metric_specification, axes_specification, filter_specifications):
    return metric_specification['filename'] + "___" + axes_specification['filename'] + "__" + "_".join(
        [filter_specification['parameter'] + "_" + str(filter_specification['value']) for filter_specification in
         filter_specifications])


def get_title_for_filter_specifications(filter_specifications):
    result = "\n".join(
        [filter_specification['parameter'] + "=" + str(filter_specification['value']) + "; " for filter_specification in
         filter_specifications])
    return result[:-2]


def extract_parameter_range(scenario_parameter_space_dict, key):
    # print "starting extraction..."
    if not isinstance(scenario_parameter_space_dict, dict):
        return None
    for generator_name, value in scenario_parameter_space_dict.iteritems():
        if generator_name == key:
            return [key], value
        if isinstance(value, list):
            if len(value) != 1:
                # print "\tdiscarding lookup of {} as this looks rather like a parameter".format(value)
                continue
            value = value[0]
            result = extract_parameter_range(value, key)
            if result is not None:
                path, values = result
                return [generator_name, 0] + path, values
        elif isinstance(value, dict):
            result = extract_parameter_range(value, key)
            if result is not None:
                path, values = result
                return [generator_name] + path, values
    # print "finishing extraction"
    return None


def extract_generation_parameters(scenario_parameter_dict, scenario_id):
    # print "starting extraction..."
    if not isinstance(scenario_parameter_dict, dict):
        return None

    results = []

    for generator_name, value in scenario_parameter_dict.iteritems():
        if isinstance(value, set) and generator_name != "all" and scenario_id in value:
            # print "returning {}".format([[generator_name]])
            return [[generator_name]]
        if isinstance(value, list):
            if len(value) != 1:
                # print "\tdiscarding lookup of {} as this looks rather like a parameter".format(value)
                continue
            value = value[0]
            result = extract_generation_parameters(value, scenario_id)
            if result is not None:
                for atomic_result in result:
                    results.append([generator_name] + atomic_result)
        elif isinstance(value, dict):
            result = extract_generation_parameters(value, scenario_id)
            if result is not None:
                for atomic_result in result:
                    results.append([generator_name] + atomic_result)

    if results == []:
        return None
    else:
        # print "returning {}".format(results)
        return results


def lookup_scenarios_having_specific_values(scenario_parameter_space_dict, path, value):
    current_path = path[:]
    current_dict = scenario_parameter_space_dict
    while len(current_path) > 0:
        if isinstance(current_path[0], basestring):
            current_dict = current_dict[current_path[0]]
            current_path.pop(0)
        elif current_path[0] == 0:
            current_path.pop(0)
    # print current_dict
    return current_dict[value]


paper_mode = True


def construct_output_path_and_filename(metric_specification, axes_specification, filter_specifications=None):
    # TODO: use base
    filter_spec_path = "no_filter"
    filter_filename = "no_filter.png"
    if filter_specifications:
        filter_spec_path, filter_filename = _construct_path_and_filename_for_filter_spec(filter_specifications)
    base = "../../../SCA-Data/plots/"
    date = strftime("%Y-%m-%d", gmtime())
    axes_filename = axes_specification['filename']
    output_path = os.path.join(base, date, axes_filename, filter_spec_path)
    filename = os.path.join(output_path, metric_specification['filename'] + "_" + filter_filename)
    return output_path, filename


def _construct_path_and_filename_for_filter_spec(filter_specifications):
    filter_path = ""
    filter_filename = ""
    for spec in filter_specifications:
        filter_path = os.path.join(filter_path, (spec['parameter'] + "_" + str(spec['value'])))
        filter_filename += spec['parameter'] + "_" + str(spec['value']) + "_"
    filter_filename = filter_filename[:-1] + ".png"
    return filter_path, filter_filename


def _construct_filter_specs(scenario_parameter_space_dict, maxdepth=3):
    parameter_filters = ["number_of_requests", "fix_leaf_mapping", "topology", "potential_nodes_factor"]
    parameter_value_dic = dict()
    for parameter in parameter_filters:
        _, parameter_values = extract_parameter_range(scenario_parameter_space_dict,
                                                      parameter)
        parameter_value_dic[parameter] = parameter_values
    # print parameter_value_dic.values()
    result_list = []
    for i in range(1, maxdepth + 1):
        for combi in combinations(parameter_value_dic, i):
            values = []
            for element_of_combi in combi:
                values.append(parameter_value_dic[element_of_combi])
            for v in product(*values):
                filter = []
                for (parameter, value) in zip(combi, v):
                    filter.append({'parameter': parameter, 'value': value})
                result_list.append(filter)
    return result_list


def plot_single_heatmap_general(dc_baseline,
                                dc_randround,
                                metric_specification,
                                axes_specification,
                                filter_specifications=None,
                                show_plot=False,
                                save_plot=True,
                                do_not_generate_plot_if_file_exists=False,
                                forbidden_scenario_ids=None):
    # data extraction
    spd = dc_baseline.scenario_parameter_container.scenario_parameter_dict
    sps = dc_baseline.scenario_parameter_container.scenarioparameter_room
    path_x_axis, xaxis_parameters = extract_parameter_range(sps, axes_specification['x_axis_parameter'])
    path_y_axis, yaxis_parameters = extract_parameter_range(sps, axes_specification['y_axis_parameter'])
    output_path, filename = construct_output_path_and_filename(metric_specification,
                                                               axes_specification,
                                                               filter_specifications)
    if do_not_generate_plot_if_file_exists and os.path.exists(filename):
        print "skipping {} as this file already exists".format(filename)
        return

    # for heatmap plot
    xaxis_parameters.sort()
    yaxis_parameters.sort()

    # all heatmap values will be stored in X
    X = np.zeros((len(yaxis_parameters), len(xaxis_parameters)))
    column_labels = yaxis_parameters
    row_labels = xaxis_parameters
    fig, ax = plt.subplots(figsize=(5, 4))

    min_number_of_observed_values = 10000000000000
    max_number_of_observed_values = 0

    for x_index, x_val in enumerate(xaxis_parameters):
        # all scenario indices which has x_val as xaxis parameter (e.g. node_resource_factor = 0.5
        x_set_indices = lookup_scenarios_having_specific_values(spd, path_x_axis, x_val)
        for y_index, y_val in enumerate(yaxis_parameters):
            y_set_indices = lookup_scenarios_having_specific_values(spd, path_y_axis, y_val)
            pos_indices = x_set_indices & y_set_indices
            # TODO: delete filter?
            if filter_specifications:
                for filter_specification in filter_specifications:
                    if (axes_specification['x_axis_parameter'] == filter_specification['parameter'] or
                                axes_specification['y_axis_parameter'] == filter_specification['parameter']):
                        continue
                    else:
                        filter_path, _ = extract_parameter_range(sps, filter_specification['parameter'])
                        filter_indices = lookup_scenarios_having_specific_values(spd, filter_path,
                                                                                 filter_specification['value'])
                        pos_indices = pos_indices & filter_indices

            if forbidden_scenario_ids:
                pos_indices = pos_indices - forbidden_scenario_ids

            solutions_by_index = [(dc_baseline.get_solutions_by_scenario_index(x)['ClassicMCF'][0],
                                   dc_randround.get_solutions_by_scenario_index(x)
                                   ['RandomizedRoundingTriumvirat'][0])
                                  for x in pos_indices if not
                                  dc_randround.get_solutions_by_scenario_index(x)
                                  ['RandomizedRoundingTriumvirat'][0] is None
                                  ]

            values = [metric_specification['lookup_function'](baseline_sol, randround_sol)
                      for (baseline_sol, randround_sol) in solutions_by_index]

            if 'metric_filter' in metric_specification:
                values = [value for value in values if metric_specification['metric_filter'](value)]

            if len(values) < min_number_of_observed_values:
                min_number_of_observed_values = len(values)
            if len(values) > max_number_of_observed_values:
                max_number_of_observed_values = len(values)

            m = np.mean(values)

            if 'rounding_function' in metric_specification:
                rounded_m = metric_specification['rounding_function'](m)
            else:
                rounded_m = float("{0:.1f}".format(round(m, 2)))

            plt.text(x_index + .25,
                     y_index + .33,
                     rounded_m,
                     fontsize=14.5,
                     color='w',
                     path_effects=[PathEffects.withStroke(linewidth=5, foreground="k")]
                     )

            X[y_index, x_index] = rounded_m

    if min_number_of_observed_values == max_number_of_observed_values:
        solution_count_string = "{} values per square".format(min_number_of_observed_values)
    else:
        solution_count_string = "between {} and {} values per square".format(min_number_of_observed_values,
                                                                             max_number_of_observed_values)

    if paper_mode:
        ax.set_title(metric_specification['name'], fontsize=17)
    else:
        ax.set_title(metric_specification['name'] + "\n\n" +
                     get_title_for_filter_specifications(filter_specifications) + "\n\n" +
                     solution_count_string)

    heatmap = ax.pcolor(X,
                        cmap=metric_specification['cmap'],
                        vmin=metric_specification['vmin'],
                        vmax=metric_specification['vmax'])

    if not paper_mode:
        fig.colorbar(heatmap, label=metric_specification['name'] + ' - mean in blue')
    else:
        cbar = fig.colorbar(heatmap)
        cbar.ax.tick_params(labelsize=15.5)
    ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(row_labels, minor=False, fontsize=15.5)
    ax.set_xlabel(axes_specification['x_axis_title'], fontsize=16)
    ax.set_ylabel(axes_specification['y_axis_title'], fontsize=16)
    ax.set_yticklabels(column_labels, minor=False, fontsize=15.5)

    plt.tight_layout()

    if save_plot:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print "saving plot: {}".format(filename)
        plt.savefig(filename)
    if show_plot:
        plt.show()

    plt.close(fig)


def compute_average_node_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x == "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_average_edge_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x != "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_max_node_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x == "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def compute_max_edge_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x != "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def compute_avg_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_max_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


"""
Collection of metrics.

"""
metric_specification_obj = dict(
    name="Obj. Gap [%]",
    lookup_function=lambda x, y: x.status.objGap * 100,
    filename="objective_gap",
    vmin=0.0,
    vmax=20.0,
    cmap="Blues",
    metric_filter=lambda obj: (obj >= -0.00001),
)

metric_specification_runtime = dict(
    name="Runtime [s]",
    lookup_function=lambda x, y: x.temporal_log.log_entries[-1].globaltime,
    filename="runtime",
    vmin=60,
    vmax=2400,
    cmap="Greys",
    rounding_function=lambda x: int(round(x))
)

metric_specification_runtime_randround_preprocessing = dict(
    name="Runtime Pre-Processing[s]",
    lookup_function=lambda x, y: y.meta_data.time_preprocessing,
    filename="randround_runtime_pre",
    vmin=0,
    vmax=60,
    cmap="Greys",
)

metric_specification_runtime_randround_optimization = dict(
    name="Runtime LP [s]",
    lookup_function=lambda x, y: y.meta_data.time_optimization,
    filename="randround_runtime_opt",
    vmin=0,
    vmax=900,
    cmap="Greys",
    rounding_function=lambda x: int(round(x))
)

metric_specification_runtime_randround_postprocessing = dict(
    name="Runtime Post-Processing [s]",
    lookup_function=lambda x,
                           y: y.meta_data.time_postprocessing,
    filename="randround_runtime_post",
    vmin=0,
    vmax=60,
    cmap="Greys",
)

metric_specification_runtime_randround_runtime = dict(
    name="Runtime RR Total [s]",
    lookup_function=lambda x, y: y.meta_data.time_preprocessing +
                                 y.meta_data.time_optimization +
                                 y.meta_data.time_postprocessing,
    filename="randround_runtime_total",
    vmin=0,
    vmax=2400,
    cmap="Greys",
)

metric_specification_runtime_mdk_runtime = dict(
    name="Runtime MDK [s]",
    lookup_function=lambda x, y: y.mdk_meta_data.time_preprocessing +
                                 y.mdk_meta_data.time_optimization +
                                 y.mdk_meta_data.time_postprocessing,
    filename="mdk_runtime_total",
    vmin=0,
    vmax=60,
    cmap="Greys",
)

metric_specification_embedding_ratio = dict(
    name="Acceptance Ratio",
    lookup_function=lambda x, y: x.embedding_ratio * 100.0,
    filename="embedding_ratio",
    vmin=0.0,
    vmax=100.0,
    cmap="Greens",
)

metric_specification_embedding_ratio_cleaned = dict(
    name="Acceptance Ratio (feasible) [%]",
    lookup_function=lambda x, y: (
                                     x.embedding_ratio * x.original_number_requests / x.nu_real_req) * 100,
    filename="cleaned_embedding_ratio",
    vmin=0.0,
    vmax=100,
    cmap="Greens",
)

metric_specification_nu_real_req = dict(
    name="#Feasible Requests",
    lookup_function=lambda x, y: x.nu_real_req,
    filename="real_req",
    vmin=0,
    vmax=125,
    cmap="Greens",
)

metric_specification_average_node_load = dict(
    name="Avg. Node Load",
    lookup_function=lambda x, y: compute_average_node_load(x),
    filename="avg_node_load",
    vmin=0.0,
    vmax=100,
    cmap="Oranges",
)

metric_specification_average_edge_load = dict(
    name="Avg. Edge Load [%]",
    lookup_function=lambda x, y: compute_average_edge_load(x),
    filename="avg_edge_load",
    vmin=0.0,
    vmax=100,
    cmap="Purples",
)

metric_specification_max_node_load = dict(
    name="Max. Node Load [%]",
    lookup_function=lambda x, y: compute_max_node_load(x),
    filename="max_node_load",
    vmin=0.0,
    vmax=100,
    cmap="Oranges",
)

metric_specification_max_edge_load = dict(
    name="Max. Edge Load [%]",
    lookup_function=lambda x, y: compute_max_edge_load(x),
    filename="max_edge_load",
    vmin=0.0,
    vmax=100,
    cmap="Purples",
)

metric_specification_max_load = dict(
    name="MaxLoad (Edge and Node)",
    lookup_function=lambda x, y: compute_max_load(x),
    filename="max_load",
    vmin=0.0,
    vmax=100,
    cmap="Reds",
)

metric_specification_avg_load = dict(
    name="AvgLoad (Edge and Node)",
    lookup_function=lambda x, y: compute_avg_load(x),
    filename="avg_load",
    vmin=0.0,
    vmax=100,
    cmap="Reds",
)

metric_specification_comparison_baseline_mdk = dict(
    name="Relative Performance\nMDK [%]",
    lookup_function=lambda x, y: (
                                     y.mdk_result.profit / x.status.objValue) * 100,
    filename="comparison_baseline_mdk",
    vmin=0.0,
    vmax=100.0,
    cmap="Reds",
)

metric_specification_comparison_baseline_wo_vio = dict(
    name="Relative Performance\nRounding w/o Augmentations",
    lookup_function=lambda x, y: (
                                     y.result_wo_violations.profit / x.status.objValue) * 100,
    filename="comparison_baseline_rr",
    vmin=0.0,
    vmax=100,
    cmap="Reds",
)

global_metric_specfications = [
    metric_specification_max_node_load,
    metric_specification_max_edge_load,
    metric_specification_obj,
    metric_specification_runtime,
    metric_specification_embedding_ratio,
    metric_specification_average_node_load,
    metric_specification_average_edge_load,
    metric_specification_max_load,
    metric_specification_avg_load,
    metric_specification_nu_real_req,
    metric_specification_embedding_ratio_cleaned,
    metric_specification_runtime_randround_preprocessing,
    metric_specification_runtime_randround_optimization,
    metric_specification_runtime_randround_postprocessing,
    metric_specification_comparison_baseline_mdk,
    metric_specification_comparison_baseline_wo_vio,
    metric_specification_runtime_randround_runtime,
    metric_specification_runtime_mdk_runtime,
]
"""
Axes specification used for the plots.
"""
axes_specification_resources = dict(
    x_axis_parameter="node_resource_factor",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Node Resource Factor",
    y_axis_title="Edge Resource Factor",
    filename="AXES_RESOURFES"
)

axes_specification_requests_edge_load = dict(
    x_axis_parameter="number_of_requests",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Number of Requests",
    y_axis_title="Edge Resource Factor",
    filename="AXES_NO_REQ_vs_EDGE_RF"
)

global_axes_specifications = [axes_specification_resources,
                              axes_specification_requests_edge_load]


def select_scenarios_with_high_objective_gap_or_zero_requests(dc_baseline, algorithm_name,
                                                              output_respective_generation_parameters=True):
    scenario_ids = dc_baseline.algorithm_scenario_solution_dictionary[algorithm_name].keys()

    result = []

    for scenario_id in scenario_ids:
        scenario_solution = dc_baseline.get_solutions_by_scenario_index(scenario_id)[algorithm_name][0]
        scenario_status = scenario_solution.status
        if scenario_status.objGap > 100:
            result.append(scenario_id)

            if output_respective_generation_parameters:
                print "Scenario {} has a very high gap, i.e. a gap of {} due to the objective bound being {} and the objective value being {}".format(
                    scenario_id,
                    scenario_status.objGap,
                    scenario_status.objBound,
                    scenario_status.objValue
                )
                print "The computation for this scenario took {} seconds.".format(scenario_solution.runtime)
                print "This scenario had the following generation parameters:"
                generation_parameters = extract_generation_parameters(
                    dc_baseline.scenario_parameter_container.scenario_parameter_dict, scenario_id
                )
                for gen_param in generation_parameters:
                    print "\t {}".format(gen_param)
        if scenario_solution.nu_real_req < 0.5:
            result.append(scenario_id)

            if output_respective_generation_parameters:
                print "Scenario {} has doesn't have any reasonable scenarios in it...{}".format(scenario_id,
                                                                                                scenario_status.objGap,
                                                                                                scenario_status.objBound,
                                                                                                scenario_status.objValue)
                print "The computation for this scenario took {} seconds.".format(scenario_solution.runtime)
                print "This scenario had the following generation parameters:"
                generation_parameters = extract_generation_parameters(
                    dc_baseline.scenario_parameter_container.scenario_parameter_dict, scenario_id
                )
                for gen_param in generation_parameters:
                    print "\t {}".format(gen_param)

    print "{} many scenarios experienced a very, very high gap or contained 0 requests".format(len(result))
    return result


def plot_bound_ecdf(reduced_pickle_baseline, reduced_pickle_rand, path_filename_tuple=None, showplot=False):
    """
    Plots for a given datacontainers an empirical distribution function for comparison of dual bounds.

    :param reduced_pickle_baseline: unpickled datacontainer of baseline experiments (e.g. MIP)
    :param reduced_pickle_rand: unpickled datacontainer of randomized rounding experiments
    :param path_filename_tuple: tuple containing save as filename in path (Default value = None)
    :param showplot: show plot after generating it (Default value = False)
    :return:
    """
    spd = reduced_pickle_baseline.scenario_parameter_container.scenario_parameter_dict
    # scenario parameter space
    sps = reduced_pickle_baseline.scenario_parameter_container.scenarioparameter_room

    scenario_ids_baseline = reduced_pickle_baseline.algorithm_scenario_solution_dictionary["ClassicMCF"].keys()

    filter_path, values = extract_parameter_range(sps, "number_of_requests")
    # print "filter path is {}".format(filter_path)


    scenario_ids_baseline = reduced_pickle_baseline.algorithm_scenario_solution_dictionary["ClassicMCF"].keys()

    result = [[] * len(values)]

    mip_bound_deviation = []

    fix, ax = plt.subplots()

    for value in values:
        filter_indices = lookup_scenarios_having_specific_values(spd, filter_path, value)
        for scenario_id in filter_indices:
            scenario_solution_baseline = reduced_pickle_baseline.get_solutions_by_scenario_index(scenario_id)["ClassicMCF"][0]
            scenario_solution_rand = reduced_pickle_rand.get_solutions_by_scenario_index(scenario_id)["RandomizedRoundingTriumvirat"][0]

            if scenario_solution_rand is None:
                # print "solution of scenario {} is None; will not consider it".format(scenario_id)
                continue
            log_time_root = 10 ** 100
            root_entry = scenario_solution_baseline.temporal_log.root_relaxation_entry
            if root_entry is not None:
                log_time_root = root_entry.globaltime

            first_log_entry = scenario_solution_baseline.temporal_log.log_entries[0]
            log_time_first_entry = first_log_entry.globaltime
            if log_time_root < log_time_first_entry:
                mip_bound = root_entry.data.objective_bound
            else:
                mip_bound = first_log_entry.data.objective_bound

            # mip_solution = scenario_solution_baseline.status.objValue

            # mip_bound_time = scenario_solution_baseline.temporal_log.log_entries[0].globaltime

            novel_bound = scenario_solution_rand.meta_data.status.objValue

            # for log_entry in scenario_solution_baseline.temporal_log.log_entries:
            #    print log_entry

            if novel_bound > 0.0:
                mip_bound_deviation.append(mip_bound / novel_bound)

            if mip_bound > 1.1 * novel_bound:
                # print "mip solution is {}".format(mip_solution)
                # print "mip bound at time {} is {}".format(mip_bound_time, mip_bound)
                # print "novel bound is {}".format(novel_bound)
                # print "\n\n"
                pass

        sorted = np.sort(np.array(mip_bound_deviation))
        yvals = np.arange(len(sorted)) / float(len(sorted))

        ax.plot(sorted, yvals, label="#requests: {}".format(value))

    ax.grid(True, which="both")

    ax.set_xlim([0.5, 9.0])
    ax.set_ylim([0, 1.0])

    ax.set_xscale("log", basex=2)
    print [x.get_text() for x in ax.get_xticklabels()]
    print [x for x in ax.get_xticklines()]

    ticks = [2 ** p for p in range(-1, 4)]
    ax.set_xticks(ticks, minor=False)
    ax.set_xticklabels(ticks, minor=False)

    yticks = [0.1, 0.3, 0.5, 0.7, 0.9]
    ax.set_yticks(yticks, minor=True)

    plt.xlabel("MIP dual bound / novel dual bound")
    plt.ylabel("ECDF")
    plt.title("Comparison of dual bounds (root linear programming relaxation)")
    plt.tight_layout()
    # plt.savefig("Comparison_of_dual_bounds.png")
    plt.legend(loc="lower right")
    if path_filename_tuple:
        path, filename = path_filename_tuple
        if not os.path.exists(path):
            os.makedirs(path)
        print "saving plot: {}".format(filename)
        plt.savefig(filename)
    if showplot:
        plt.show()


def scatter_obj_load(reduced_pickle_baseline, reduced_pickle_rand, saveplots=False):
    spd = reduced_pickle_baseline.scenario_parameter_container.scenario_parameter_dict
    # scenario parameter space
    sps = reduced_pickle_baseline.scenario_parameter_container.scenarioparameter_room

    scenario_ids_baseline = reduced_pickle_baseline.algorithm_scenario_solution_dictionary["ClassicMCF"].keys()

    filter_path, _ = extract_parameter_range(sps, "number_of_requests")
    # print "filter path is {}".format(filter_path)
    filter_indices = lookup_scenarios_having_specific_values(spd, filter_path, 50)

    x = [[], [], [], []]
    y = [[], [], [], []]
    labels = ["min. augmentation",
              "max. profit",
              "randomized rounding (without any capacity violations)",
              "multi-dimensional knapsack"]
    markers = ['o', 'v', 'x', '+']
    colors = ['r', 'g', 'b', 'k']

    for scenario_id in scenario_ids_baseline:
        scenario_solution_baseline = reduced_pickle_baseline.get_solutions_by_scenario_index(scenario_id)["ClassicMCF"][
            0]
        scenario_solution_rand = reduced_pickle_rand.get_solutions_by_scenario_index(scenario_id)["RandomizedRoundingTriumvirat"][0]
        if scenario_solution_rand is None:
            print "solution of scenario {} is None; will not consider it".format(scenario_id)
            continue

        # print scenario_solution_rand
        # print "\n\n"

        mip_solution = scenario_solution_baseline.status.objValue

        first_mip_solution = None
        for log_entry in scenario_solution_baseline.temporal_log.log_entries:
            if log_entry.data.solution_count > 0 and log_entry.data.objective_value > 0.01 and log_entry.globaltime < 2500:
                first_mip_solution = log_entry.data.objective_value

        if first_mip_solution is None:
            first_mip_solution = 1

        if mip_solution < 0.01:
            continue

        data = [scenario_solution_rand.collection_of_samples_with_violations[0],
                scenario_solution_rand.collection_of_samples_with_violations[1],
                scenario_solution_rand.result_wo_violations,
                scenario_solution_rand.mdk_result]

        for i, dat in enumerate(data):
            x[i].append((dat.profit / first_mip_solution) * 100)
        for i, dat in enumerate(data):
            y[i].append((max(dat.max_node_load, dat.max_edge_load) * 100))

    fig, ax = plt.subplots(figsize=(9, 4))

    for i in range(1, -1, -1):
        ax.scatter(x[i], y[i], c=colors[i], marker=markers[i], label=labels[i], s=10, linewidths=.1, alpha=.6)

    ax.set_xlim([25, 400])
    ax.set_ylim([0, 600])
    plt.grid(True, which="both")
    # plt.xticks([25, 5, 75, 100, 200, 400])
    # plt.xticklabels(["0.25", "0.5", "0.75", "1.0", "2.0", "4.0"])

    # plt.yticks([50, 100, 200, 400, 600])
    # plt.yticklabels(["0.5", "1.0", "2.0", "4.0", "6.0"])

    plt.xscale("log")

    if False:
        ticklines = ax.get_xticklines() + ax.get_yticklines()
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

        for line in ticklines:
            line.set_linewidth(3)

        for line in gridlines:
            line.set_linestyle('-.')

        for label in ticklabels:
            label.set_color('r')
            label.set_fontsize('medium')

    xticks = [50, 100, 200, 300, ]
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(xticks, minor=False, fontsize=15.5)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15.5)
    ax.set_xlim((45, 350))
    ax.set_ylim((0, 550))

    plt.legend(loc="upper left", fontsize=16)

    ax.set_title("Vanilla Randomized Rounding Performance", fontsize=17)
    ax.set_ylabel("maximal load [%]", fontsize=16)
    ax.set_xlabel("profit relative to baseline solution [%]", fontsize=16)
    plt.tight_layout()
    if saveplots:
        output_path, filename = construct_output_path_and_filename({'filename': 'ECDF_scatter'},
                                                                   {'filename': 'generalplot'})
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print "saving plot: {}".format(filename)
        plt.savefig(filename)

    fix, ax = plt.subplots()

    sorted_x = [[], [], [], []]
    for i in range(len(x)):
        sorted_x[i] = np.sort(x[i])

    yvals = np.arange(len(sorted_x[0])) / float(len(sorted_x[0]))

    for i in range(len(x)):
        ax.plot(sorted_x[i], yvals, label=labels[i])

    plt.grid(True, which="both")

    plt.legend(loc="upper left")
    ax.set_title("ECDF of empirical Approximation Ratios")
    ax.set_xlabel("achieved profit w.r.t. MIP solution (100% means equal profit)")
    ax.set_ylabel("ECDF")
    plt.tight_layout()
    if saveplots:
        output_path, filename = construct_output_path_and_filename({'filename': 'ECDF_profigit'},
                                                                   {'filename': 'generalplot'})
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print "saving plot: {}".format(filename)
        plt.savefig(filename)

    fix, ax = plt.subplots()

    sorted_y = [[], [], [], []]
    for i in range(len(x)):
        sorted_y[i] = np.sort(y[i])

    for i in range(0, 2):
        ax.plot(sorted_y[i], yvals, label=labels[i])

    plt.legend(loc="upper left")
    ax.set_title("ECDF of Capacity Violations")
    ax.set_xlabel("Maximally Observed Capacity Violations")
    ax.set_ylabel("ECDF")
    ticks = [2 ** p for p in range(-3, 3)]
    ax.set_xticks(ticks, minor=False)
    ax.set_xticklabels(ticks, minor=False)
    ax.grid(True, which="major")

    plt.tight_layout()
    if saveplots:
        output_path, filename = construct_output_path_and_filename({'filename': 'ECDF_capacity_violations'},
                                                                   {'filename': 'generalplot'})
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print "saving plot: {}".format(filename)
        plt.savefig(filename)


def plot_heatmaps(dc_baseline,
                  dc_randround,
                  metric_specifications=global_metric_specfications,
                  axes_specifications=global_axes_specifications,
                  show_plot=False,
                  save_plot=True,
                  do_not_generate_plot_if_file_exists=False,
                  forbidden_scenario_ids=None,
                  plot_ecdf=True,
                  plot_scatter=True,
                  maxdepthfilter=3):
    """ Main function for evaluation, creating plots and saving them in a specific directory hierarchy. Each metric
    specification and each axes specification combination will be an heatmap plot. Additionally permutation of length 3
    with all scenarioparametergeneration will be generated and fixed for plotting. This leads to the directory hierarchy.
    For examples number_of_req_25 will be the directory which containing plots where parameter number_of_req is fixed
    to 25. ECDF and scatter plots will be also plotted as default.

    :param dc_baseline: unpickled datacontainer of baseline experiments (e.g. MIP)
    :param dc_randround: unpickled datacontainer of randomized rounding experiments
    :param metric_specifications: metric specifications used by plots (Default value = global_metric_specifications)
    :param axes_specifications: each axes specifications used as x/y axis in plots (Default value = global_metric_specifications)
    :param show_plot: show plot after generating it (Default value = False)
    :param save_plot: save plots to given structure (Default value = False)
    :param do_not_generate_plot_if_file_exists: False if you want to overwrite existing plots (Default value = False)
    :param forbidden_scenario_ids: List of scenario ids which should NOT be included in evaluation (Default value = None)
    :param plot_ecdf: plot ECDF plots (Default = True)
    :param plot_scatter: plot scatter plots (Default = True)
    :param maxdepthfilter: Permutation legnth for scenarioparametergeneration (Default = 3)
    :return:
    """
    if plot_ecdf:
        path_filename_tuple = construct_output_path_and_filename({'filename': 'ecdf_plot'}, {'filename': 'generalplot'})
        plot_bound_ecdf(dc_baseline, dc_randround, path_filename_tuple)

    if plot_scatter:
        scatter_obj_load(dc_baseline, dc_randround, save_plot)

    filter_specs = _construct_filter_specs(dc_baseline.scenario_parameter_container.scenarioparameter_room, maxdepth=maxdepthfilter)
    for metric_specfication in metric_specifications:
        for axes_speci in axes_specifications:
            plot_single_heatmap_general(dc_baseline,
                                        dc_randround,
                                        metric_specfication,
                                        axes_speci,
                                        show_plot=show_plot,
                                        save_plot=save_plot,
                                        do_not_generate_plot_if_file_exists=do_not_generate_plot_if_file_exists,
                                        forbidden_scenario_ids=forbidden_scenario_ids
                                        )
            for filter_spec in filter_specs:
                plot_single_heatmap_general(dc_baseline,
                                            dc_randround,
                                            metric_specfication,
                                            axes_speci,
                                            filter_specifications=filter_spec,
                                            show_plot=False,
                                            save_plot=save_plot,
                                            do_not_generate_plot_if_file_exists=False,
                                            )
