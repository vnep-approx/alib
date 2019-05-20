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
            finds a path containing the subfolders "input", "output", "log",
            indicating that this is the root folder of the experiment.
            ALIB_DIR is then set to the parent of this directory.

        INPUT_DIR:
            This is the path containing any configuration files, scenario pickle files
            and similar files, as provided in the optional "extras" argument in
            deployment.

        LOG_DIR:
            This is the path, where log files related to the experiment should be stored.

        OUTPUT_DIR:
            This is the path where the results of this execution should be stored
            results.

    LOG_DIR and OUTPUT_DIR are required to be empty to avoid accidental overwriting of previous
    results.
    """

    ALIB_DIR = None

    LOG_DIR = "./log"
    INPUT_DIR = None
    OUTPUT_DIR = None

    def __init__(self):
        raise Exception("Run ExperimentPathHandler.initialize() and then access ExperimentPathHandler.LOG_DIR etc.")

    @staticmethod
    def initialize(check_emptiness_output=True, check_emptiness_log=True):
        log.info("Initializing Paths...")
        ExperimentPathHandler.ALIB_DIR = ExperimentPathHandler._get_alib_dir()
        ExperimentPathHandler.EXPERIMENT_DIR = ExperimentPathHandler._get_experiment_dir()

        ExperimentPathHandler.LOG_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR , "log")
        ExperimentPathHandler.INPUT_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR, "input")
        ExperimentPathHandler.OUTPUT_DIR = os.path.join(ExperimentPathHandler.EXPERIMENT_DIR, "output")

        _experiment_paths = {ExperimentPathHandler.LOG_DIR,
                             ExperimentPathHandler.INPUT_DIR,
                             ExperimentPathHandler.OUTPUT_DIR}

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
        if check_emptiness_output and not ExperimentPathHandler._is_empty(ExperimentPathHandler.OUTPUT_DIR):
            raise AlibPathError("Experiment output path is not empty: {}".format(
                ExperimentPathHandler.OUTPUT_DIR
            ))
        if check_emptiness_log and not ExperimentPathHandler._is_empty(ExperimentPathHandler.LOG_DIR):
            raise AlibPathError("Experiment log path is not empty: {}".format(
                ExperimentPathHandler.LOG_DIR
            ))

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
            potential_exp_dir = os.getcwd()
            print "current dir: {}".format(potential_exp_dir)
            parent = None
            iterations = 0
            while potential_exp_dir and potential_exp_dir != parent:
                parent = os.path.split(potential_exp_dir)[0]
                print "current dir: {}, parent: {}".format(potential_exp_dir, parent)
                print "{}".format(set(os.listdir(potential_exp_dir)))
                if ({"log", "input", "output"}.issubset(set(os.listdir(potential_exp_dir)))):
                    log.info("Setting alib path according to first parent containing only input, log, output and sca folders")
                    alib_dir = potential_exp_dir
                    break
                iterations += 1
                potential_exp_dir = parent

                if iterations >= 100:
                    raise AlibPathError("Exceeded iterations {}".format(iterations))

        if alib_dir is None or not os.path.isdir(alib_dir):
            raise AlibPathError("Invalid alib root path: {}".format(alib_dir))

        log.info("Alib root dir:       {}".format(alib_dir))
        return os.path.abspath(alib_dir)

    @staticmethod
    def _get_experiment_dir():
        #for legacy purposes
        return ExperimentPathHandler._get_alib_dir()


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
    ''' Custom implementation of a pretty printer to output classes nicely.

    '''

    _DESCRIBED_HERE = 0
    _DESCRIBED_ABOVE = 1
    _DESCRIBED_BELOW = 2

    def __init__(self, indent_offset=0, indent_step=1, whitelist=None, max_depth=10000):
        self.base_indent_offset = indent_offset
        self.indent_step = indent_step
        if whitelist is None:
            whitelist = ["request", "substrate", "scenario", "graph", "scenariosolution", "request_generator", "linearrequest", "treewidth_model_experiments"]
        self.whitelist = whitelist
        self.max_depth = max_depth

        self._known_objects = None

    def pprint(self, obj, col_output_limit=None):
        """
        Generate a string representation of the object and its attributes.

        :param obj: A python object
        :param col_output_limit: maximum collection limit
        :return: pretty printed string
        """
        # if not PrettyPrinter._has_class_and_module(obj):  # fallback in case obj has undefined class/module attributes
        #     return str(obj)
        self._known_objects = set()
        result = "\n".join(self._generate_lines(obj, col_output_limit=col_output_limit))
        return result

    def _generate_lines(self, obj, depth=0, col_output_limit=None):
        if not col_output_limit:
            col_output_limit = sys.maxint
        self._objects_to_explore = [obj]
        while self._objects_to_explore:
            obj = self._objects_to_explore.pop()
            if depth > self.max_depth:
                print "Maximum depth exceeded!"
                continue

            children = []
            # if self._is_known(obj):
            #     continue
            self._remember(obj)
            yield self._basic_object_description(obj, depth)

            # what kind of type is the object?
            if isinstance(obj, dict):
                if not obj:
                    header = self._get_header(depth + 1)
                    yield "{header} empty dict\n".format(header=header)
                line = ""
                for index, (key, value) in enumerate(obj.iteritems()):
                    if index < col_output_limit:
                        if self._has_class_and_module(value):
                            header = self._get_header(depth + 1)
                            line = "{head}{key}:\n".format(head=header, key=key)
                            for bla in self._generate_lines(value, depth=depth + 1, col_output_limit=col_output_limit):
                                line += bla
                        elif isinstance(value, (list, tuple, set, dict)):
                            header = self._get_header(depth + 1)
                            line = "{head}{key}:\n".format(head=header, key=key)
                            if len(value) > 1:
                                for bla in self._generate_lines(value, depth=depth + 1,  col_output_limit=col_output_limit):
                                    line += bla
                            else:
                                header = self._get_header(depth + 2)
                                line += "{head}{value}\n".format(head=header, value=value)
                        else:
                            header = self._get_header(depth + 1)
                            line = "{head}{key}: {value}\n".format(head=header, key=key, value=value)
                    else:
                        header = self._get_header(depth + 1)
                        line = "{header}[ ... ]\n".format(header=header)
                        yield line
                        break
                    yield line
            if isinstance(obj, (list, tuple, set)):
                header = self._get_header(depth + 1)
                line = "{head}{obj}\n".format(head=header, obj=obj)
                for index, elem in enumerate(obj):
                    if index < col_output_limit:
                        if self._has_class_and_module(elem):
                            for bla in self._generate_lines(elem, depth=depth + 1,  col_output_limit=col_output_limit):
                                line += bla
                        elif isinstance(elem, (list, tuple, set, dict)):
                            for bla in self._generate_lines(elem, depth=depth + 1,  col_output_limit=col_output_limit):
                                line += bla
                    else:
                        line += "{header}[ ... ]\n".format(header=header)
                        break
                yield line

            if self._has_class_and_module(obj):
                for attr in sorted(obj.__dict__.keys()):
                    value = obj.__dict__[attr]
                    if isinstance(value, (list, tuple, set, dict)):
                            header = self._get_header(depth + 1)
                            line = "{head}{attr}:\n".format(head=header, attr=attr)
                            for bla in self._generate_lines(value, depth=depth + 1, col_output_limit=col_output_limit):
                                line += bla
                    else:
                        header = self._get_header(depth + 1)
                        line = "{head}{key}: {value}\n".format(head=header, key=attr, value=value)
                    yield line

    def _basic_object_description(self, obj, depth):
        header = self._get_header(depth)
        if self._has_class_and_module(obj):
            line = "{head}{mod_class} @ {id}\n".format(head=header,
                                                       mod_class=self._get_module_and_class(obj),
                                                       id=hex(id(obj)))
        else:
            line =""
        return line

    def _get_relative_position_of_description(self, value):
        position = PrettyPrinter._DESCRIBED_HERE
        if PrettyPrinter._has_class_and_module(value):
            if self._matches_whitelist(value):
                if self._is_known(value):
                    position = PrettyPrinter._DESCRIBED_ABOVE
                else:
                    position = PrettyPrinter._DESCRIBED_BELOW
        return position

    def _get_basic_attribute_description(self, attr, value, pos, depth):
        header = self._get_header(depth + 1)
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
        print "match?\n{}\n{}".format(value, mod_class)
        # print "matches...", mod_class, self.whitelist, mod_class.split(".")
        return any(allowed_module in mod_class.split(".") for allowed_module in self.whitelist)

    def _is_known(self, obj):
        return id(obj) in self._known_objects

    def _remember(self, obj):
        self._known_objects.add(id(obj))


    @staticmethod
    def _has_class_and_module(value):
        return hasattr(value, "__class__") and hasattr(value, "__module__")


def pretty_print(obj, indentOffset=0, indentStep=2, whitelist=None, max_depth=10000):
    return PrettyPrinter(indent_offset=indentOffset,
                         indent_step=indentStep,
                         whitelist=whitelist,
                         max_depth=max_depth).pprint(obj)


def initialize_root_logger(filename, print_level=logging.INFO, file_level=logging.DEBUG, allow_override=False):
    if not allow_override and filename is not None and os.path.exists(filename):
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


def get_logger_filename(logger_name):
    return os.path.join(ExperimentPathHandler.LOG_DIR, logger_name + ".log")

def get_logger(logger_name, make_file=True, make_stream=False, print_level=logging.INFO, file_level=logging.DEBUG, propagate=True, allow_override=False):
    logger = logging.getLogger(logger_name)

    if len(logger.handlers) == 0:
        if make_file:
            fname = get_logger_filename(logger_name)
            if not allow_override and os.path.exists(fname):
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

    for node in graph.nodes:
        graphviz_lines.append("{} ;".format(node))

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
