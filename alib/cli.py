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


@cli.command()
@click.argument('dc_baseline', type=click.File('r'))
@click.argument('dc_randround', type=click.File('r'))
def full_evaluation(dc_baseline, dc_randround):
    baseline_data = pickle.load(dc_baseline)
    randround_data = pickle.load(dc_randround)
    print "data loaded"
    evaluation.plot_heatmaps(baseline_data, randround_data)


def f_generate_scenarios(scenario_output_file, parameters, threads):
    click.echo('Generate Scenarios')
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(parameters.name).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_scenario_generation.log".format(file_basename))
    util.initialize_root_logger(log_file)
    scenariogeneration.generate_pickle_from_yml(parameters, scenario_output_file, threads)


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
