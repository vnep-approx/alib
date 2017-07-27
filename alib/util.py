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

import collections
import logging
import os
import shutil
import sys
import time
from random import Random

random = Random("_util")


class DeploymentError(Exception): pass


class RangeError(Exception): pass


class AlibPathError(Exception): pass


class ExperimentPathHandler(object):
    """
    This class handles the paths related to the execution of experiments or
    the generation of scenarios using alib.

    To use it, run ExperimentPathHandler.initialize() to initialize the
    variables described below:

        ALIB_DIR:
            This is the parent folder for any experiments to be run on the server.

            It is equal to the environment variable ALIB_EXPERIMENT_HOME if it is set.

            Otherwise the code traverses up through the file system
            starting at this file's location (paths.py) until it
            finds a path containing the subfolders "input", "output", "log"
            and "sca", indicating that this is the root folder of the experiment.
            ALIB_DIR is then set to the parent of this directory.

            Finally, if no matching alib folder can be found, the default is
            to use the relative path "../../..".

        EXPERIMENT_DIR:
            This is the directory containing the "input", "output", "log" and "sca"
            folders for the current experiment. It is found by

        INPUT_DIR:
            This is the path containing any configuration files, scenario pickle files
            and similar files, as provided in the optional "extras" argument in
            deployment.

        LOG_DIR:
            This is the path, where log files related to the experiment should be stored.

        OUTPUT_DIR:
            This is the path where the results of this execution should be stored
            results.

        CODE_DIR:
            This is the path containing the deployed codebase. This path is added to
            the system path variable to enable importing modules from the entire codebase.

    LOG_DIR and OUTPUT_DIR are required to be empty to avoid accidental overwriting of previous
    results.
    """

    CURRENT_FILE_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    ALIB_DIR = None

    EXPERIMENT_DIR = None

    LOG_DIR = "./log"
    INPUT_DIR = None
    OUTPUT_DIR = None
    CODE_DIR = None

    def __init__(self):
        raise Exception("Run ExperimentPathHandler.initialize() and then access ExperimentPathHandler.LOG_DIR etc.")

    @staticmethod
    def initialize():
        log.info("Initializing Paths...")
        ExperimentPathHandler.ALIB_DIR = ExperimentPathHandler._get_alib_dir()
        ExperimentPathHandler.EXPERIMENT_DIR = ExperimentPathHandler._get_experiment_dir()

        ExperimentPathHandler.LOG_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR, "log")
        ExperimentPathHandler.INPUT_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR, "input")
        ExperimentPathHandler.OUTPUT_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR, "output")
        ExperimentPathHandler.CODE_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR, "sca")

        _experiment_paths = {ExperimentPathHandler.LOG_DIR,
                             ExperimentPathHandler.INPUT_DIR,
                             ExperimentPathHandler.OUTPUT_DIR,
                             ExperimentPathHandler.CODE_DIR}

        # Check that all experiment paths exist:
        if not all(os.path.exists(p) for p in _experiment_paths):
            raise AlibPathError("Path(s) do not exist:\n    " + "\n   ".join(
                p for p in _experiment_paths if not os.path.exists(p)
            ))

        # Check that all of these paths are proper directories
        if not all(os.path.isdir(p) for p in _experiment_paths):
            raise AlibPathError("Could not find path(s):\n    " + "\n   ".join(
                p for p in _experiment_paths if not os.path.isdir(p)
            ))

        # Check that output & log folders are empty, to avoid overwriting previous results:
        if not ExperimentPathHandler._is_empty(ExperimentPathHandler.OUTPUT_DIR):
            raise AlibPathError("Experiment output path is not empty: {}".format(
                ExperimentPathHandler.OUTPUT_DIR
            ))

        if ExperimentPathHandler.CODE_DIR not in sys.path:
            log.info("Adding alib code directory to system path variable: {}".format(
                ExperimentPathHandler.CODE_DIR)
            )
            sys.path.append(ExperimentPathHandler.CODE_DIR)

    @staticmethod
    def _is_empty(path):
        return len(os.listdir(path)) == 0

    @staticmethod
    def _get_alib_dir():
        alib_dir = None
        if os.getenv("ALIB_EXPERIMENT_HOME") is not None:
            log.info("Setting path according to ALIB_EXPERIMENT_HOME")
            alib_dir = os.getenv("ALIB_EXPERIMENT_HOME")
        else:
            log.info("ALIB_EXPERIMENT_HOME environment variable not found")
            potential_exp_dir = ExperimentPathHandler.CURRENT_FILE_DIR
            parent = os.path.split(potential_exp_dir)[0]
            iterations = 0
            while potential_exp_dir and potential_exp_dir != parent:
                potential_exp_dir = parent
                parent = os.path.split(potential_exp_dir)[0]
                if ({"log", "input", "output", "sca", "gurobi.log"} == set(os.listdir(potential_exp_dir)) or
                            {"log", "input", "output", "sca"} == set(os.listdir(potential_exp_dir))):
                    log.info("Setting alib path according to first parent containing only input, log, output and sca folders")
                    alib_dir = parent
                    break
                iterations += 1

            if iterations >= 100:
                raise AlibPathError("Exceeded iterations")
        if alib_dir is None:
            log.info("Setting alib path to default, three levels above current file")
            alib_dir = os.path.abspath(os.path.join(ExperimentPathHandler.CURRENT_FILE_DIR, '../../..'))

        if not os.path.isdir(alib_dir):
            raise AlibPathError("Invalid alib root path: {}".format(alib_dir))

        log.info("Alib root dir:       {}".format(alib_dir))
        return os.path.abspath(alib_dir)

    @staticmethod
    def _get_experiment_dir():
        parent, child = os.path.split(ExperimentPathHandler.CURRENT_FILE_DIR)
        iterations = 0
        while parent != ExperimentPathHandler.ALIB_DIR and iterations < 100:
            parent, child = os.path.split(parent)
            iterations += 1
        if iterations >= 100:
            raise AlibPathError("Exceeded iterations")
        exp_dir = os.path.abspath(os.path.join(parent, child))

        log.info("Experiment root dir: {}".format(exp_dir))

        if not os.path.isdir(exp_dir):
            raise AlibPathError("Invalid experiment path: {}".format(exp_dir))
        return os.path.abspath(exp_dir)


class CodebaseDeployer(object):
    """
    This class simplifies deployment of the code.

    """

    def __init__(self,
                 code_base_id,
                 remote_base_dir,
                 extra=None,
                 local_base_path=".",
                 servers=None,
                 export_path="deployment",
                 cleanup=True):

        if servers is None:
            servers = ["localhost"]
        self.servers = servers
        self.number_of_servers = len(servers)
        self.local_base_path = os.path.abspath(local_base_path)
        self.export_directory = os.path.abspath(os.path.join(local_base_path, export_path))
        self.remote_base_dir = remote_base_dir
        self.cleanup = cleanup
        self.code_base_id = code_base_id
        self._generated_files = set()
        self.extra = extra

    def deploy_codebase(self):
        """
        Make a copy of the codebase and deploy it on the servers.
        :return:
        """
        self._create_paths_on_remote()
        self._create_codebase_snapshot()
        self._deploy_code_to_servers()
        if self.extra:
            self._deploy_extra_to_server()
        if self.cleanup:
            self._cleanup()

    def _create_codebase_snapshot(self):
        def ignored_files(path, names):  # see https://docs.python.org/2/library/shutil.html#shutil.copytree
            if os.path.abspath(path) == self.export_directory or ".git" in path or ".idea" in path:
                return names
            print "\t", path
            include = [".gml", ".py"]
            only_files = (name for name in names if os.path.isfile(os.path.join(path, name)))
            ignored = [name for name in only_files
                       if not any(name.endswith(ext) for ext in include)]
            return ignored

        code_source_directory = self.local_base_path
        code_export_directory = os.path.join(self.export_directory, self.code_base_id)
        if os.path.exists(code_export_directory):
            raise DeploymentError("The export directory exists!")
        code_subdir_name = "sca"
        src_dir = os.path.join(code_export_directory, code_subdir_name)
        shutil.copytree(code_source_directory, src_dir, ignore=ignored_files)
        tar = shutil.make_archive(code_export_directory,
                                  format="gztar",
                                  root_dir=self.export_directory,
                                  base_dir=os.path.join(self.code_base_id, code_subdir_name))
        if self.cleanup:
            shutil.rmtree(code_export_directory)
        self._generated_files.add(tar)

    def _deploy_code_to_servers(self):
        tar_name = self.code_base_id + ".tar.gz"
        for server in self.servers:
            scp_command = "scp {tar_file} {server}:{remote_code_dir}"
            command = scp_command.format(tar_file=os.path.join(self.export_directory, tar_name),
                                         server=server,
                                         remote_code_dir=os.path.join(self.remote_base_dir, tar_name))
            self._execute_command(command)
            extract_tar_command = 'ssh {server} "cd {remote_code_dir} ; tar -xzvf {tar_name}"'
            command = extract_tar_command.format(server=server,
                                                 remote_code_dir=self.remote_base_dir,
                                                 tar_name=tar_name)
            self._execute_command(command)

    def _deploy_extra_to_server(self):
        for server in self.servers:
            for filename in self.extra:
                scp_command = "scp {extra_file} {server}:{remote_code_dir}"
                command = scp_command.format(extra_file=os.path.abspath(filename.name),
                                             server=server,
                                             remote_code_dir=os.path.join(self.remote_base_dir, self.code_base_id,
                                                                          "input/",
                                                                          os.path.basename(filename.name)))
                self._execute_command(command)

    def _create_paths_on_remote(self):
        subdir = ['input/', 'output/', 'log/']
        for server in self.servers:
            mkdir_command = "ssh {} \"mkdir {}\"".format(server, os.path.join(self.remote_base_dir, self.code_base_id))
            self._execute_command(mkdir_command)
            for sd in subdir:
                mkdir_command = "ssh {} \"mkdir {}\"".format(server, os.path.join(self.remote_base_dir, self.code_base_id, sd))
                self._execute_command(mkdir_command)

    def _cleanup(self):
        if self.cleanup:
            print "Starting Cleanup..."
            for f in self._generated_files:
                f_path = os.path.abspath(f)
                print "Deleting", f_path
                if os.path.isdir(f):
                    print "Cleanup: omitting directory {}".format(f)
                    continue
                if " " in f or "\t" in f or "\n" in f:
                    print "Aborting cleanup, filename contains spaces: {}".format(f)
                    break
                if not f_path.startswith(os.path.abspath(self.export_directory)):
                    print "Aborting cleanup, file seems outside of export directory: {}".format(f)
                    break
                os.remove(f)
            self._generated_files = set()

    def _execute_command(self, command):
        print command
        os.system(command)


class PrintLogger(object):
    @staticmethod
    def debug(message, *args, **kwargs):
        print message

    @staticmethod
    def info(message, *args, **kwargs):
        print message

    @staticmethod
    def warning(message, *args, **kwargs):
        print message

    @staticmethod
    def error(message, *args, **kwargs):
        print message

    @staticmethod
    def critical(message, *args, **kwargs):
        print message

    @staticmethod
    def log(message, *args, **kwargs):
        print message


class PrettyPrinter(object):
    _DESCRIBED_HERE = 0
    _DESCRIBED_ABOVE = 1
    _DESCRIBED_BELOW = 2

    def __init__(self, indent_offset=0, indent_step=2, whitelist=None, max_depth=10000):
        self.base_indent_offset = indent_offset
        self.indent_step = indent_step
        if whitelist is None:
            whitelist = ["request", "substrate", "scenario", "graph", "scenariosolution", "request_generator", "linearrequest"]
        self.whitelist = whitelist
        self.max_depth = max_depth

        self._current_depth = 0
        self._depth = None
        self._known_objects = None

    def pprint(self, obj):
        """
        Generate a string representation of the object and its attributes.

        :param obj: A python object
        :return: pretty printed string
        """
        if not PrettyPrinter._has_class_and_module(obj):  # fallback in case obj has undefined class/module attributes
            return str(obj)
        self._known_objects = set()
        result = "\n".join(self._generate_lines(obj))
        return result

    def _generate_lines(self, obj):
        self._objects_to_explore = [obj]
        self._depth = [0]
        while self._objects_to_explore:
            obj = self._objects_to_explore.pop()
            self._current_depth = self._depth.pop()
            if self._current_depth > self.max_depth:
                print "Maximum depth exceeded!"
                continue

            children = []
            # if self._is_known(obj):
            #     continue
            self._remember(obj)
            yield self._basic_object_description(obj)

            # explore the object's attributes:
            for attr in sorted(obj.__dict__.keys()):
                value = obj.__dict__[attr]
                position = self._get_relative_position_of_description(value)
                if position == PrettyPrinter._DESCRIBED_BELOW:
                    children.append(value)
                line = self._get_basic_attribute_description(attr, value, position)

                if isinstance(value, collections.Iterable) and not isinstance(value, basestring):
                    contained_objects = self._get_objects_from_iterable(value)
                    if contained_objects:
                        line += " [see above or below]"
                    children.extend(o for o in contained_objects if not self._is_known(o))
                yield line

            # queue any new children for exploration
            if children:
                while children:
                    child = children.pop()
                    self._add_to_exploration_list(child)

    def _basic_object_description(self, obj):
        header = self._get_header(self._current_depth)
        line = "\n{head}{mod_class} @ {id}".format(head=header,
                                                   mod_class=self._get_module_and_class(obj),
                                                   id=hex(id(obj)))
        return line

    def _get_relative_position_of_description(self, value):
        position = PrettyPrinter._DESCRIBED_HERE
        print value
        if PrettyPrinter._has_class_and_module(value):
            print "value is class or module"
            if self._matches_whitelist(value):
                print "is on whitelist"
                if self._is_known(value):
                    position = PrettyPrinter._DESCRIBED_ABOVE
                else:
                    position = PrettyPrinter._DESCRIBED_BELOW
        return position

    def _get_basic_attribute_description(self, attr, value, pos):
        header = self._get_header(self._current_depth + 1)
        if pos == PrettyPrinter._DESCRIBED_HERE:
            line = "{head}{attr}: {value}".format(head=header, attr=attr, value=value)
        elif pos == PrettyPrinter._DESCRIBED_ABOVE:
            line = "{head}{attr}: {value} [see above]".format(head=header, attr=attr, value=value)
        elif pos == PrettyPrinter._DESCRIBED_BELOW:
            line = "{head}{attr}: {value} [see below]".format(head=header, attr=attr, value=value)
        else:
            raise Exception("invalid description position!")
        return line

    def _get_objects_from_iterable(self, value):
        contained_objects = []
        for element in value:
            if isinstance(value, dict):
                element = value[element]
            if PrettyPrinter._has_class_and_module(element):
                if self._matches_whitelist(element):
                    contained_objects.append(element)
        return contained_objects

    def _get_module_and_class(self, value):
        return "{mod}.{c}".format(mod=value.__module__,
                                  c=value.__class__.__name__)

    def _get_header(self, current_depth):
        header = " " * (self.base_indent_offset + self.indent_step * current_depth)
        return header

    def _matches_whitelist(self, value):
        mod_class = self._get_module_and_class(value)
        print "matches...", mod_class, self.whitelist, mod_class.split(".")
        return any(allowed_module in mod_class.split(".") for allowed_module in self.whitelist)

    def _is_known(self, obj):
        return id(obj) in self._known_objects

    def _remember(self, obj):
        self._known_objects.add(id(obj))

    def _add_to_exploration_list(self, obj):
        self._objects_to_explore.append(obj)
        self._depth.append(self._current_depth + 1)

    @staticmethod
    def _has_class_and_module(value):
        return hasattr(value, "__class__") and hasattr(value, "__module__")


def pretty_print(obj, indentOffset=0, indentStep=2, whitelist=None, max_depth=10000):
    return PrettyPrinter(indent_offset=indentOffset,
                         indent_step=indentStep,
                         whitelist=whitelist,
                         max_depth=max_depth).pprint(obj)


def initialize_root_logger(filename, print_level=logging.INFO, file_level=logging.DEBUG):
    if filename is not None and os.path.exists(filename):
        raise AlibPathError("Attempted to overwrite existing log file:  {}".format(filename))
    print "Initializing root logger: ", filename
    fmt = '%(levelname)-10s %(asctime)s %(lineno)4d:%(name)-32s\t %(message)s'
    logging.basicConfig(filename=filename,
                        filemode='w',
                        level=file_level,
                        format=fmt)

    root_logger = logging.getLogger()

    # # This code breaks in weird ways:
    # file_handler = logging.FileHandler(filename, mode="w")
    # file_handler.setLevel(file_level)
    # file_handler.setFormatter(fmt)
    # root_logger.addHandler(file_level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(print_level)
    root_logger.addHandler(stdout_handler)
    root_logger.info("Initialized Root Logger")


def get_logger(logger_name, make_file=True, make_stream=False, print_level=logging.INFO, file_level=logging.DEBUG, propagate=True):
    logger = logging.getLogger(logger_name)

    if len(logger.handlers) == 0:
        if make_file:
            fname = os.path.join(ExperimentPathHandler.LOG_DIR, logger_name + ".log")
            if os.path.exists(fname):
                raise AlibPathError("Attempted to overwrite existing log file:  {}".format(fname))
            file_handler = logging.FileHandler(fname, mode="w")
            file_handler.setLevel(file_level)
            formatter = logging.Formatter(fmt='%(levelname)-10s %(asctime)s %(lineno)4d:%(name)-32s\t %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        if make_stream:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(print_level)
            logger.addHandler(stdout_handler)
        logger.propagate = propagate
        logger.debug("Created logger {}".format(logger_name))

    return logger


def log_start_and_end_of_function(logger=PrintLogger, start_message="Start: {f}({args})", end_message="End:   {f}({args}) after {t} s."):
    def decorator(f):
        def log_start_end(*args, **kwargs):
            arg_string = ""
            if "{args}" in start_message:
                arg_string = ", ".join(str(arg) for arg in args)
                if kwargs:
                    arg_string += ", "
                    arg_string += ", ".join("{}={}".format(key, arg) for key, arg in kwargs.iteritems())
            logger.info(start_message.format(f=f.__name__, args=arg_string))
            start_time = time.time()
            result = f(*args, **kwargs)

            duration = time.time() - start_time
            logger.info(end_message.format(f=f.__name__, args=arg_string, t=duration))
            return result

        return log_start_end

    return decorator


def check_percentage(value, none_allowed=True):
    if none_allowed and value is None:
        return
    if not isinstance(value, float):
        raise TypeError("Expected float, got value of type {}".format(type(value)))
    if value < 0.0 or value > 1.0:
        raise RangeError("Float {} should be between 0.0 and 1.0".format(value))


def check_positive(value, none_allowed=True):
    if none_allowed and value is None:
        return
    if not isinstance(value, (int, long, float)):
        raise TypeError("Expected number, got {}".format(type(value)))
    if value < 0.0:
        raise RangeError("Expected positive number, got {}".format(value))


def check_int(value, none_allowed=True):
    if none_allowed and value is None:
        return
    if not isinstance(value, int):
        raise Exception("Bad Type")
    if value < 0:
        raise Exception("Bad Type")


def check_within_range(value, min, max, none_allowed=True):
    if none_allowed and value is None:
        return
    if not isinstance(value, (int, float)):
        raise TypeError("Expected number, got {}".format(type(value)))
    if not min <= value <= max:
        raise RangeError("Expected number within range {} - {}, got {}".format(min, max, value))


def check_bool(value, none_allowed=True):
    if none_allowed and value is None:
        return
    if not isinstance(value, bool):
        raise TypeError("Expected boolean, got {}".format(type(value)))


def approx_equal(x, y, accuracy=0.0001):
    return abs(x - y) < accuracy


def get_obj_gap(objective, bound):
    if objective < -10 ** 99 or bound > 10 ** 99:
        return 10.0 ** 100
    if objective != 0:
        return abs(abs(bound) - abs(objective)) / abs(objective)
    if abs(bound - objective) < 0.0001:
        return 0.0
    return 10.0 ** 100


def get_graph_viz_string(graph, directed=True, get_edge_style=lambda e: ""):
    graphviz_lines = []
    if directed:
        graphviz_lines.append("digraph {} {{".format(graph.name))
        edge_symbol = "->"
    else:
        graphviz_lines.append("graph {} {{".format(graph.name))
        edge_symbol = "--"
    for edge in sorted(graph.edges):
        n1, n2 = edge
        graphviz_lines.append("  {} {} {} [{}];".format("\"{}\"".format(n1), edge_symbol, "\"{}\"".format(n2), get_edge_style((n1, n2))))
    graphviz_lines.append("}\n")
    gv = "\n".join(graphviz_lines)
    return gv


def graph_viz_edge_color_according_to_request_list(edge_set_list, colors=None):
    if colors is None:
        colors = ["\"#{:02x}{:02x}{:02x}\"".format(random.randrange(256),
                                                   random.randrange(256),
                                                   random.randrange(256))
                  for _ in edge_set_list]

    def inner(e):
        set_list_index = edge_set_list.index(next(edges for edges in edge_set_list if e in edges))
        return "color={}".format(colors[set_list_index])

    return inner


if __name__ == "__main__":
    log = initialize_root_logger(None, logging.CRITICAL, logging.DEBUG)
else:
    log = get_logger(__name__, make_file=False, propagate=True)
