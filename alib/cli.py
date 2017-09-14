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

import os
import pickle

import click

from . import evaluation, run_experiment, scenariogeneration, solutions, util

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions


@click.group()
def cli():
    pass


@cli.command()
@click.argument('codebase_id')
@click.argument('remote_base_dir', type=click.Path())
@click.option('--local_base_dir', type=click.Path(exists=True), default=".")
@click.argument('servers')
@click.option('--extra', '-e', multiple=True, type=click.File())
def deploy_code(codebase_id, remote_base_dir, local_base_dir, servers, extra):
    f_deploy_code(codebase_id, remote_base_dir, local_base_dir, servers, extra)


def f_deploy_code(codebase_id, remote_base_dir, local_base_dir, servers, extra):
    """
    Deploys the codebase on a remote server.

    This function is separated from deploy_code so that it can be reused when extending the CLI from outside the alib.

    :param codebase_id:
    :param remote_base_dir:
    :param local_base_dir:
    :param servers:
    :param extra:
    :return:
    """
    click.echo('Deploy codebase')
    if not local_base_dir:
        local_base_dir = os.path.abspath("../../")
    local_base_dir = os.path.abspath(local_base_dir)
    deployer = util.CodebaseDeployer(
        code_base_id=codebase_id,  # string, some unique codebase id
        remote_base_dir=remote_base_dir,  # server path object specifying paths for codebase
        local_base_path=local_base_dir,  # local root directory of the codebase
        servers=servers.split(","),  # servers to deploy to
        cleanup=True,  # delete auto-generated files?
        extra=extra
    )
    deployer.deploy_codebase()


@cli.command()
@click.argument('scenario_output_file')
@click.argument('parameters', type=click.File('r'))
@click.option('--threads', default=1)
def generate_scenarios(scenario_output_file, parameters, threads):
    f_generate_scenarios(scenario_output_file, parameters, threads)


def f_generate_scenarios(scenario_output_file, parameter_file, threads):
    """
    Generates the scenarios according to the scenario parameters found in the parameter_file.

    This function is separated from generate_scenarios so that it can be reused when extending the CLI from outside
    the alib.

    :param scenario_output_file: path to pickle file to which the resulting scenarios will be written
    :param parameter_file: readable file object containing the scenario parameters in yml format
    :param threads: number of concurrent threads used for scenario generation
    :return:
    """
    click.echo('Generate Scenarios')
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(parameter_file.name).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_scenario_generation.log".format(file_basename))
    util.initialize_root_logger(log_file)
    scenariogeneration.generate_pickle_from_yml(parameter_file, scenario_output_file, threads)


@cli.command()
@click.argument('dc_baseline', type=click.File('r'))
@click.argument('dc_randround', type=click.File('r'))
def full_evaluation(dc_baseline, dc_randround):
    baseline_data = pickle.load(dc_baseline)
    randround_data = pickle.load(dc_randround)
    print "data loaded"
    evaluation.plot_heatmaps(baseline_data, randround_data)


@cli.command()
@click.argument('experiment_yaml', type=click.File('r'))
@click.argument('min_scenario_index', type=click.INT)
@click.argument('max_scenario_index', type=click.INT)
@click.option('--concurrent', default=1)
def start_experiment(experiment_yaml,
                     min_scenario_index, max_scenario_index,
                     concurrent):
    f_start_experiment(experiment_yaml,
                       min_scenario_index, max_scenario_index,
                       concurrent)


def f_start_experiment(experiment_yaml,
                       min_scenario_index, max_scenario_index,
                       concurrent):
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
    util.initialize_root_logger(log_file)

    run_experiment.run_experiment(
        experiment_yaml,
        min_scenario_index, max_scenario_index,
        concurrent
    )


if __name__ == '__main__':
    cli()
