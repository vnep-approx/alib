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

import sys
import time
import traceback
from collections import namedtuple

import gurobipy
from gurobipy import GRB, LinExpr

from . import datamodel, util


class ModelcreatorError(Exception): pass


class TemporalLogError(Exception): pass


class SolutionError(Exception): pass


class AlgorithmResult(object):
    ''' Abstract Algorithm result only specifying specific functions and no data storage capabilities.

    '''

    def get_solution(self):
        '''

        :return: the solution (as a namedtuple) stored in this class; abstract function
        '''
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def cleanup_references(self, original_scenario):
        self._check_scenarios_are_equal(original_scenario)
        self._cleanup_references_raw(original_scenario)

    def _check_scenarios_are_equal(self, original_scenario):
        ''' Checks whether the scenario stored within this result equals the one handed over.

        :param original_scenario:
        :return:
        '''
        errors = []
        solution = self.get_solution()
        own_scenario = solution.scenario

        # some sanity checks:
        if len(original_scenario.requests) != len(own_scenario.requests):
            errors.append("Mismatched number of requests, expected {}, found {}".format(
                len(original_scenario.requests),
                len(own_scenario.requests)
            ))
        for original_request, own_req in zip(original_scenario.requests, own_scenario.requests):
            if original_request.nodes != own_req.nodes:
                errors.append("Mismatched node sets in requests {}, {}".format(original_request.name, own_req.name))
            if original_request.edges != own_req.edges:
                errors.append("Mismatched edge sets in requests {}, {}".format(original_request.name, own_req.name))
        if original_scenario.substrate.name != own_scenario.substrate.name:
            errors.append("Mismatched substrates: {}, {}".format(original_scenario.substrate.name, own_scenario.substrate.name))
        if errors:
            raise SolutionError("Errors in cleanup of solution {}:\n   - {}".format(
                self,
                "\n   - ".join(errors)
            ))

    def _cleanup_references_raw(self, original_scenario):
        ''' Abstract function used to cleanup references. In particular, this abstract function can be used to replace
            references to objects stored inside the original scenario. This is useful as per default instances are
            executed using multiple processes (multiprocessing) and accordingly, returned solutions do not reference
            the original substrate graph. Essentially, implementing this function in a meaningful way, storage space
            in the returned pickles is saved.

        :param original_scenario:
        :return:
        '''
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")


def gurobi_callback(model, where):
    ''' A guronbi callback used to log the temporal progress during the exection process of gurobi.

    :param model: the gurobi model from which the callback is executed
    :param where: code referencing for what reason (where in the execution) the callback is executed.
    :return:
    '''
    try:
        mc = model._mc
        if where == GRB.callback.POLLING:
            pass
        elif where == GRB.callback.MIPNODE:
            if mc.temporal_log.root_relaxation_entry is None:
                t = model.cbGet(GRB.callback.RUNTIME)
                nodecnt = model.cbGet(GRB.callback.MIPNODE_NODCNT)
                objbst = model.cbGet(GRB.callback.MIPNODE_OBJBST)
                objbnd = model.cbGet(GRB.callback.MIPNODE_OBJBND)
                solcnt = model.cbGet(GRB.callback.MIPNODE_SOLCNT)
                mc.temporal_log.set_root_relaxation_entry(MIPData(nodecnt, objbst, objbnd, solcnt, GRB.callback.MIPNODE), t)
        elif where == GRB.callback.MIP:
            t = model.cbGet(GRB.callback.RUNTIME)
            nodecnt = model.cbGet(GRB.callback.MIP_NODCNT)
            objbst = model.cbGet(GRB.callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.callback.MIP_OBJBND)
            solcnt = model.cbGet(GRB.callback.MIP_SOLCNT)
            mc.temporal_log.add_log_data(MIPData(nodecnt, objbst, objbnd, solcnt, GRB.callback.MIP), t)
        elif where == GRB.callback.MIPSOL:
            t = model.cbGet(GRB.callback.RUNTIME)
            nodecnt = model.cbGet(GRB.callback.MIPSOL_NODCNT)
            objbst = model.cbGet(GRB.callback.MIPSOL_OBJBST)
            objbnd = model.cbGet(GRB.callback.MIPSOL_OBJBND)
            solcnt = model.cbGet(GRB.callback.MIPSOL_SOLCNT)
            # print "\nMIPSOL CALLBACK: ", "nodecnt:", nodecnt, "objbst:", objbst, "objbnd:", objbnd, "solcnt:", solcnt
            mc.temporal_log.add_log_data(MIPData(nodecnt, objbst, objbnd, solcnt, GRB.callback.MIPSOL), t)
        elif where == GRB.callback.MESSAGE:
            for line in model.cbGet(GRB.callback.MSG_STRING).rstrip('\n').split("\n"):
                mc.logger.debug(line)

    except Exception:
        print sys.exc_info()[0]
        traceback.print_exc()


def build_construct_name(spec):
    """
    Build a construct_name function.

    This is used e.g. to construct the LP variable names.

    The ``spec`` parameter defines a list of argument names of the resulting
    name constructor.

    The resulting name constructor is a function with one positional argument
    (``name``) and keyword arguments defined in ``spec``. The constructed name
    starts with the ``name`` arguments and then contains the given keyword
    arguments in the order of ``spec``.

    Keyword arguments are formatted as ``"_prefix[value]"`` where the prefix
    is the argument key if the prefix itself is unset or ``None``. If the
    argument has a custom formatter, it is called on the value.

    Example 1:

    >>> construct_name = build_construct_name([
    ...     ("req_name", "req"),
    ...     "vnode",
    ...     "snode",
    ...     "vedge",
    ...     "sedge",
    ... ])
    ...
    >>> construct_name(
    ...     "node_mapping",
    ...     req_name="req1",
    ...     vnode="i",
    ...     snode="u",
    ... )
    ...
    "node_mapping_req[req1]_vnode[i]_snode[u]"

    Example 2:

    >>> construct_name = build_construct_name([
    ...     ("bag", None, lambda v: "_".join(sorted(v))),
    ... ])
    ...
    >>> construct_name(
    ...     "name",
    ...     bag={"i", "j", "k"}
    ... )
    ...
    "name_bag[i_j_k]"

    :param spec:
        list of argument names or tuples of ``(argument name, prefix, formatter)``,
        where trailing ``None`` values can be left out
    :return: construct_name function
    """

    def noop_formatter(v):
        return v

    extended_spec = []
    for arg in spec:
        key = prefix = formatter = None
        if isinstance(arg, str):
            key = arg
            formatter = noop_formatter
        elif isinstance(arg, tuple) and 1 <= len(arg) <= 3:
            if len(arg) >= 1:
                key = arg[0]
                if not isinstance(key, str):
                    raise TypeError("argument name must be str")
            if len(arg) >= 2:
                prefix = arg[1]
            if len(arg) == 3:
                formatter = arg[2] or noop_formatter
        else:
            raise TypeError("spec must be a list of strings or tuples with min length 1 and max length 3")
        extended_spec.append((key, prefix or key, formatter or noop_formatter))

    def _construct_name(name, **kwargs):
        for key, prefix, formatter in extended_spec:
            if key in kwargs:
                value = kwargs.pop(key)
                name += "_{}[{}]".format(prefix, formatter(value))
        if kwargs:
            raise TypeError("{}() got an unexpected keyword argument '{}'".format(
                _construct_name.__name__, kwargs.keys()[0]))
        return name.replace(" ", "")

    _construct_name.spec = extended_spec

    return _construct_name


construct_name = build_construct_name([
    ("req_name", "req"),
    "type", "vnode", "snode", "vedge", "sedge", "other",
    ("sub_name", "substrate"),
    ("sol_name", "solution"),
])

Param_MIPGap = "MIPGap"
Param_IterationLimit = "IterationLimit"
Param_NodeLimit = "NodeLimit"
Param_Heuristics = "Heuristics"
Param_Threads = "Threads"
Param_TimeLimit = "TimeLimit"
Param_MIPFocus = "MIPFocus"
Param_RootCutPasses = "CutPasses"
Param_Cuts = "Cuts"
Param_NodefileStart = "NodefileStart"
Param_NodeMethod = "NodeMethod"
Param_Method = "Method"
Param_BarConvTol = "BarConvTol"
Param_NumericFocus = "NumericFocus"
Param_LogToConsole = "LogToConsole"
Param_Crossover = "Crossover"


def isFeasibleStatus(status):
    result = True
    if status == GurobiStatus.INFEASIBLE:
        result = False
    elif status == GurobiStatus.INF_OR_UNBD:
        result = False
    elif status == GurobiStatus.UNBOUNDED:
        result = False
    elif status == GurobiStatus.LOADED:
        result = False

    return result


class GurobiStatus(object):

    ''' Represents the status information of Gurobi after its execution.

    In particular, this class stores Gurobi's status code, the solution count, the objective value, the objective bound,
    the objective gap and whether an integral solution was computed.
    '''

    LOADED = 1  # Model is loaded, but no solution information is available.
    OPTIMAL = 2  # Model was solved to optimality (subject to tolerances), and an optimal solution is available.
    INFEASIBLE = 3  # Model was proven to be infeasible.
    INF_OR_UNBD = 4  # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.
    UNBOUNDED = 5  # Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.
    CUTOFF = 6  # Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available.
    ITERATION_LIMIT = 7  # Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.
    NODE_LIMIT = 8  # Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.
    TIME_LIMIT = 9  # Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.
    SOLUTION_LIMIT = 10  # Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.
    INTERRUPTED = 11  # Optimization was terminated by the user.
    NUMERIC = 12  # Optimization was terminated due to unrecoverable numerical difficulties.
    SUBOPTIMAL = 13  # Unable to satisfy optimality tolerances; a sub-optimal solution is available.
    IN_PROGRESS = 14  # A non-blocking optimization call was made (by setting the NonBlocking parameter to 1 in a Gurobi Compute Server environment), but the associated optimization run is not yet complete.

    def __init__(self,
                 status=1,
                 solCount=0,
                 objValue=GRB.INFINITY,
                 objBound=GRB.INFINITY,
                 objGap=GRB.INFINITY,
                 integralSolution=True
                 ):
        self.solCount = solCount
        self.status = status
        self.objValue = objValue
        self.objBound = objBound
        self.objGap = objGap
        self.integralSolution = integralSolution

    def _convertInfinityToNone(self, value):
        if value is GRB.INFINITY:
            return None
        return value

    def isIntegralSolution(self):
        return self.integralSolution

    def getObjectiveValue(self):
        return self._convertInfinityToNone(self.objValue)

    def getObjectiveBound(self):
        return self._convertInfinityToNone(self.objBound)

    def getMIPGap(self):
        return self._convertInfinityToNone(self.objGap)

    def hasFeasibleStatus(self):
        return isFeasibleStatus(self.status)

    def isFeasible(self):
        feasibleStatus = self.hasFeasibleStatus()
        result = feasibleStatus
        if not self.integralSolution and feasibleStatus:
            # feasible status just means that the infeasibility wasn't proven..
            # we need to check some additional coniditions:
            return (self.status == GurobiStatus.OPTIMAL) or (self.status == GurobiStatus.SUBOPTIMAL)
        elif self.integralSolution:
            result = self.solCount > 0
            if result and not feasibleStatus:
                raise ModelcreatorError("Solutions exist, but the status ({}) indicated an infeasibility.".format(self.status))
            return result

        return result

    def isOptimal(self):
        if self.status == self.OPTIMAL:
            return True
        else:
            return False

    def __str__(self):
        return "solCount: {0}; status: {1}; objValue: {2}; objBound: {3}; objGap: {4}; integralSolution: {5}; ".format(self.solCount, self.status, self.objValue, self.objBound, self.objGap, self.integralSolution)


class GurobiSettings(object):
    ''' Represents parameter settings for gurobi.

    '''
    def __init__(self,
                 mipGap=None,
                 iterationlimit=None,
                 nodeLimit=None,
                 heuristics=None,
                 threads=None,
                 timelimit=None,
                 MIPFocus=None,
                 rootCutPasses=None,
                 cuts=None,
                 BarConvTol=None,
                 OptimalityTol=None,
                 Presolve=None,
                 nodefilestart=None,
                 method=None,
                 nodemethod=None,
                 numericfocus=None,
                 crossover=None,
                 logtoconsole=0):
        util.check_positive(mipGap)
        self.MIPGap = mipGap

        util.check_positive(iterationlimit)
        self.IterationLimit = iterationlimit

        util.check_positive(nodeLimit)
        util.check_int(nodeLimit)
        self.NodeLimit = nodeLimit

        util.check_percentage(heuristics)
        self.Heuristics = heuristics

        util.check_positive(threads)
        util.check_int(threads)
        self.Threads = threads

        util.check_positive(timelimit)
        self.TimeLimit = timelimit

        util.check_within_range(MIPFocus, 0, 2)
        util.check_int(MIPFocus)
        self.MIPFocus = MIPFocus

        self.rootCutPasses = rootCutPasses
        self.cuts = cuts

        self.BarConvTol = BarConvTol
        self.OptimalityTol = OptimalityTol
        self.Presolve = Presolve

        util.check_positive(nodefilestart)
        self.NodefileStart = nodefilestart

        self.Method = method
        self.NodeMethod = nodemethod

        util.check_within_range(numericfocus, 0, 3)
        util.check_int(numericfocus)
        self.NumericFocus = numericfocus

        util.check_within_range(crossover, 0, 4)
        self.Crossover = crossover

        util.check_within_range(logtoconsole,0,1)
        self.LogToConsole = logtoconsole

    def setTimeLimit(self, newTimeLimit):
        util.check_positive(newTimeLimit)
        self.TimeLimit = newTimeLimit

    def __str__(self):
        return str(vars(self))


class AbstractModelCreator(object):

    ''' Abstract basis for classes creating Mixed-Integer or Linear Programming models.
    Subclass this for creating Gurobi models.

    Provides essential functionality as well as a structured way to create the model and measure the time
    needed to create, execute and post-process the model.

    '''

    _listOfUserVariableParameters = [Param_MIPGap, Param_IterationLimit, Param_NodeLimit, Param_Heuristics,
                                     Param_Threads, Param_TimeLimit, Param_Cuts, Param_MIPFocus, Param_RootCutPasses,
                                     Param_NodefileStart, Param_Method, Param_NodeMethod, Param_BarConvTol, Param_NumericFocus,
                                     Param_Crossover, Param_LogToConsole]

    def __init__(self,
                 gurobi_settings=None,
                 optimization_callback=gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        self.gurobi_settings = gurobi_settings

        # callback used when gurobi's optimize()-function is called
        self.optimization_callback = optimization_callback

        # if a filename is specified the lp is output
        self.lp_output_file = lp_output_file

        # if a filename is specified and gurobi couldn't generate a feasible solution the IIS is computed and saved under
        # the specified filename
        # NOTE: computing the IIS may be really time intensive, so please use this carefully!
        self.potential_iis_filename = potential_iis_filename

        self.model = None  # the model of gurobi
        self.status = None  # GurobiStatus instance
        self.solution = None  # either a integral solution or a fractional one

        self.temporal_log = TemporalLog()

        self.time_preprocess = None
        self.time_optimization = None
        self.time_postprocessing = None

        self._time_postprocess_start = None

        if logger is None:
            self.logger = util.get_logger(__name__, make_file=False, propagate=True)
        else:
            self.logger = logger

        self._disable_temporal_information_output = False
        self._disable_temporal_log_output = False


    def init_model_creator(self):
        ''' Initializes the modelcreator by generating the model. Afterwards, model.compute() can be called to let
            Gurobi solve the model.

        :return:
        '''

        time_preprocess_start = time.clock()

        self.model = gurobipy.Model("test")
        self.model._mc = self

        self.model.setParam("LogToConsole", 0)

        if self.gurobi_settings is not None:
            self.apply_gurobi_settings(self.gurobi_settings)

        self.preprocess_input()

        self.create_variables()

        # for making the variables accessible
        self.model.update()

        self.create_constraints()

        self.create_objective()
        self.model.update()

        self.time_preprocess = time.clock() - time_preprocess_start

    def preprocess_input(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def create_variables(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def create_constraints(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def create_objective(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def compute_integral_solution(self):
        ''' Abstract function computing an integral solution to the model (generated before).

        :return: Result of the optimization consisting of an instance of the GurobiStatus together with a result
                 detailing the solution computed by Gurobi.
        '''
        self.logger.debug("Computing integral solution.")
        # do the optimization
        time_optimization_start = time.clock()
        self.model.optimize(self.optimization_callback)

        self.time_optimization = time.clock() - time_optimization_start

        #the following shall not be counted to any runtime
        if self.lp_output_file is not None:
            self.model.write(self.lp_output_file)

        # do the postprocessing
        self._time_postprocess_start = time.clock()
        gurobi_status = self.model.getAttr("Status")
        objVal = None
        objBound = GRB.INFINITY
        objGap = GRB.INFINITY
        solutionCount = self.model.getAttr("SolCount")
        nodecnt = self.model.getAttr("NodeCount")

        try:
            if solutionCount > 0:
                objVal = self.model.getAttr("ObjVal")
                # interestingly, MIPGap and ObjBound cannot be accessed when there are no variables or the MIP is infeasible..
                objGap = self.model.getAttr("MIPGap")

            if isFeasibleStatus(gurobi_status):
                objBound = self.model.getAttr("ObjBound")
        except Exception as e:
            self.logger.error("Problem accessing Gurobi Values: {}".format(e))


        if solutionCount > 0:
            self.temporal_log.add_log_data(MIPData(nodecnt, objVal, objBound, solutionCount, -1),
                                           self.model.getAttr("Runtime"),
                                           force_new_entry=True)

        if not self._disable_temporal_log_output:
            self.logger.debug("Temporal log entries:")
            self.logger.debug("    Root Relaxation Entry: {}".format(self.temporal_log.root_relaxation_entry))
            for entry in self.temporal_log.log_entries:
                self.logger.debug("    {}".format(entry))
            self.logger.debug("    Improvement Entries:")
            for entry in self.temporal_log.improved_entries:
                self.logger.debug("        {}".format(entry))

        self.status = GurobiStatus(status=gurobi_status,
                                   solCount=solutionCount,
                                   objValue=objVal,
                                   objGap=objGap,
                                   objBound=objBound,
                                   integralSolution=True)

        self.logger.debug("Found solution with status {}".format(self.status))

        result = None
        if self.status.isFeasible():
            self.solution = self.recover_integral_solution_from_variables()
            result = self.post_process_integral_computation()
        elif self.potential_iis_filename is not None:
            self.model.computeIIS()
            self.model.write(self.potential_iis_filename)

        self.time_postprocessing = time.clock() - self._time_postprocess_start

        if not self._disable_temporal_information_output:
            self.logger.debug("Preprocessing time:   {}".format(self.time_preprocess))
            self.logger.debug("Optimization time:    {}".format(self.time_optimization))
            self.logger.debug("Postprocessing time:  {}".format(self.time_postprocessing))

        return result

    def recover_integral_solution_from_variables(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def post_process_integral_computation(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def compute_fractional_solution(self):
        ''' Assuming that the original model was a Mixed-Integer Program, this function relaxes the integrality conditions
            on variables and solves the corresponding LP using Gurobi.

        :return:    GurobiStatus together with a class corresponding to the solution computed in the LP
        '''

        time_additional_preprocessing_start = time.clock()
        self.relax_model()
        self.time_preprocess = self.time_preprocess + (time.clock() - time_additional_preprocessing_start)

        # do the optimization
        time_optimization_start = time.clock()
        self.model.optimize(self.optimization_callback)
        self.time_optimization = time.clock() - time_optimization_start

        # do the postprocessing
        self._time_postprocess_start = time.clock()

        status = self.model.getAttr("Status")
        objVal = None
        objBound = GRB.INFINITY
        objGap = GRB.INFINITY
        solutionCount = self.model.getAttr("SolCount")

        if solutionCount > 0:
            objVal = self.model.getAttr("ObjVal")

        self.status = GurobiStatus(status=status,
                                   solCount=solutionCount,
                                   objValue=objVal,
                                   objGap=objGap,
                                   objBound=objBound,
                                   integralSolution=False)

        if self.lp_output_file is not None:
            self.model.write(self.lp_output_file)

        result = None
        if self.status.isFeasible():
            self.solution = self.recover_fractional_solution_from_variables()
            result = self.post_process_fractional_computation()
        elif self.potential_iis_filename is not None:
            self.model.computeIIS()
            self.model.write(self.potential_iis_filename)

        self.time_postprocessing = time.clock() - self._time_postprocess_start

        if not self._disable_temporal_information_output:
            self.logger.debug("Preprocessing time:   {}".format(self.time_preprocess))
            self.logger.debug("Optimization time:    {}".format(self.time_optimization))
            self.logger.debug("Postprocessing time:  {}".format(self.time_postprocessing))

        return result

    def relax_model(self):
        for var in self.model.getVars():
            if var.VType == GRB.BINARY:
                var.VType = GRB.CONTINUOUS
            elif var.VType == GRB.INTEGER:
                var.VType = GRB.CONTINUOUS
            elif var.VType == GRB.CONTINUOUS:
                # Continuous Variables are Fine
                pass
            else:
                raise ModelcreatorError("Cannot handle Variable Type " + str(var.VType))

    def recover_fractional_solution_from_variables(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def post_process_fractional_computation(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    ###
    ###     GUROBI SETTINGS
    ###

    def apply_gurobi_settings(self, gurobiSettings):
        ''' Apply gurobi settings.

        :param gurobiSettings:
        :return:
        '''
        if gurobiSettings.MIPGap is not None:
            self.set_gurobi_parameter(Param_MIPGap, gurobiSettings.MIPGap)
        else:
            self.reset_gurobi_parameter(Param_MIPGap)

        if gurobiSettings.IterationLimit is not None:
            self.set_gurobi_parameter(Param_IterationLimit, gurobiSettings.IterationLimit)
        else:
            self.reset_gurobi_parameter(Param_IterationLimit)

        if gurobiSettings.NodeLimit is not None:
            self.set_gurobi_parameter(Param_NodeLimit, gurobiSettings.NodeLimit)
        else:
            self.reset_gurobi_parameter(Param_NodeLimit)

        if gurobiSettings.Heuristics is not None:
            self.set_gurobi_parameter(Param_Heuristics, gurobiSettings.Heuristics)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.Threads is not None:
            self.set_gurobi_parameter(Param_Threads, gurobiSettings.Threads)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.TimeLimit is not None:
            self.set_gurobi_parameter(Param_TimeLimit, gurobiSettings.TimeLimit)
        else:
            self.reset_gurobi_parameter(Param_TimeLimit)

        if gurobiSettings.MIPFocus is not None:
            self.set_gurobi_parameter(Param_MIPFocus, gurobiSettings.MIPFocus)
        else:
            self.reset_gurobi_parameter(Param_MIPFocus)

        if gurobiSettings.cuts is not None:
            self.set_gurobi_parameter(Param_Cuts, gurobiSettings.cuts)
        else:
            self.reset_gurobi_parameter(Param_Cuts)

        if gurobiSettings.rootCutPasses is not None:
            self.set_gurobi_parameter(Param_RootCutPasses, gurobiSettings.rootCutPasses)
        else:
            self.reset_gurobi_parameter(Param_RootCutPasses)

        if gurobiSettings.NodefileStart is not None:
            self.set_gurobi_parameter(Param_NodefileStart, gurobiSettings.NodefileStart)
        else:
            self.reset_gurobi_parameter(Param_NodefileStart)

        if gurobiSettings.Method is not None:
            self.set_gurobi_parameter(Param_Method, gurobiSettings.Method)
        else:
            self.reset_gurobi_parameter(Param_Method)

        if gurobiSettings.NodeMethod is not None:
            self.set_gurobi_parameter(Param_NodeMethod, gurobiSettings.NodeMethod)
        else:
            self.reset_gurobi_parameter(Param_NodeMethod)

        if gurobiSettings.BarConvTol is not None:
            self.set_gurobi_parameter(Param_BarConvTol, gurobiSettings.BarConvTol)
        else:
            self.reset_gurobi_parameter(Param_BarConvTol)

        if gurobiSettings.NumericFocus is not None:
            self.set_gurobi_parameter(Param_NumericFocus, gurobiSettings.NumericFocus)
        else:
            self.reset_gurobi_parameter(Param_NumericFocus)

        if gurobiSettings.Crossover is not None:
            self.set_gurobi_parameter(Param_Crossover, gurobiSettings.Crossover)
        else:
            self.reset_gurobi_parameter(Param_Crossover)

        if gurobiSettings.LogToConsole is not None:
            self.set_gurobi_parameter(Param_LogToConsole, gurobiSettings.LogToConsole)
        else:
            self.reset_gurobi_parameter(Param_LogToConsole)

    def reset_all_parameters_to_default(self):
        for param in self._listOfUserVariableParameters:
            (name, type, curr, min, max, default) = self.model.getParamInfo(param)
            self.model.setParam(param, default)

    def reset_gurobi_parameter(self, param):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Parameter {} unchanged".format(param))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        self.model.setParam(param, default)

    def set_gurobi_parameter(self, param, value):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Changed value of parameter {} to {}".format(param, value))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        if not param in self._listOfUserVariableParameters:
            raise ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.setParam(param, value)

    def getParam(self, param):
        if not param in self._listOfUserVariableParameters:
            raise ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.getParam(param)


class AbstractEmbeddingModelCreator(AbstractModelCreator):
    ''' Abstract model creator designed specifically to tackle the Virtual Network Embedding Problem.
        Subclass this for more specific LPs dealing with VNEP.
        In particular, this class extends the AbstractModelCreator by instantiating some (generally needed) variables
        and generating appropriate constraints. Furthermore, it adds support for different objectives.
    '''
    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        super(AbstractEmbeddingModelCreator, self).__init__(gurobi_settings=gurobi_settings,
                                                            optimization_callback=optimization_callback,
                                                            lp_output_file=lp_output_file,
                                                            potential_iis_filename=potential_iis_filename,
                                                            logger=logger)

        self.scenario = scenario
        self.substrate = datamodel.SubstrateX(scenario.substrate)
        self.requests = scenario.requests
        self.objective = scenario.objective

        # Some variables that exist in all modelcreators
        self.var_embedding_decision = {}
        self.var_request_load = {}

    def preprocess_input(self):
        pass

    def create_variables(self):
        # these variables are the same across all model creators
        self.create_variables_embedding_decision()
        self.create_variables_request_load()

        # this abstract method allows the child-class to add its respective variables
        self.create_variables_other_than_embedding_decision_and_request_load()

        # update the model
        self.model.update()

    def create_variables_embedding_decision(self):
        for req in self.requests:
            variable_name = construct_name("embedding_decision", req_name=req.name)
            self.var_embedding_decision[req] = self.model.addVar(lb=0.0,
                                                                 ub=1.0,
                                                                 obj=0.0,
                                                                 vtype=GRB.BINARY,
                                                                 name=variable_name)

    def create_variables_request_load(self):
        for req in self.requests:
            self.var_request_load[req] = {}
            for (x, y) in self.substrate.substrate_resources:
                variable_name = construct_name("load", req_name=req.name, other=(x, y))
                self.var_request_load[req][(x, y)] = self.model.addVar(lb=0.0,
                                                                       ub=GRB.INFINITY,
                                                                       obj=0.0,
                                                                       vtype=GRB.CONTINUOUS,
                                                                       name=variable_name)

    def create_variables_other_than_embedding_decision_and_request_load(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def create_constraints(self):
        self.create_constraints_bound_node_and_edge_load_by_capacities()

        self.create_constraints_other_than_bounding_loads_by_capacities()

    def create_constraints_bound_node_and_edge_load_by_capacities(self):
        for x, y in self.substrate.substrate_resources:
            load_expr = LinExpr([(1.0, self.var_request_load[req][(x, y)]) for req in self.requests])
            constr_name = construct_name("bound_node_load_by_capacity", type=(x, y))
            self.model.addConstr(load_expr, GRB.LESS_EQUAL, self.substrate.substrate_resource_capacities[(x, y)], name=constr_name)

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def plugin_constraint_embed_all_requests(self):
        for req in self.requests:
            expr = LinExpr()
            expr.addTerms(1.0, self.var_embedding_decision[req])

            constr_name = construct_name("embed_all_requests", req_name=req.name)

            self.model.addConstr(expr, GRB.EQUAL, 1.0, name=constr_name)

    def create_objective(self):
        if self.objective == datamodel.Objective.MAX_PROFIT:
            self.plugin_objective_maximize_profit()
        elif self.objective == datamodel.Objective.MIN_COST:
            self.plugin_objective_minimize_cost()
            self.plugin_constraint_embed_all_requests()
        else:
            raise ModelcreatorError("Invalid objective type {}".format(self.objective))

    def plugin_objective_maximize_profit(self):
        objExpr = gurobipy.LinExpr()
        for req in self.requests:
            objExpr.addTerms(req.profit, self.var_embedding_decision[req])
        self.model.setObjective(objExpr, GRB.MAXIMIZE)

    def plugin_objective_minimize_cost(self):
        costlist = []
        for req in self.requests:
            for u, v in self.substrate.substrate_edge_resources:
                costlist.append(self.var_request_load[req][(u, v)]
                                * self.substrate.get_edge_cost((u, v)))
            for ntype, snode in self.substrate.substrate_node_resources:
                costlist.append(self.var_request_load[req][(ntype, snode)]
                                * self.substrate.get_node_type_cost(snode, ntype))

        obj = gurobipy.quicksum(costlist)
        self.model.setObjective(obj, GRB.MINIMIZE)


LPData = namedtuple("LPData", ["iteration_count", "objective_bound"])

MIPData = namedtuple("MIPData", ["node_count",
                                 "objective_value",
                                 "objective_bound",
                                 "solution_count",
                                 "callback_code"])

LogEntry = namedtuple("LogEntry", ["globaltime",
                                   "time_within_gurobi",
                                   "data"])


class TemporalLog(object):
    ''' Class detailing the solution process of Gurobi during its execution.

        Data is (should) be added to this class during the gurobi callback.

    '''
    def __init__(self, log_interval_in_seconds=30.0):
        self.log_entries = []
        self.improved_entries = []
        self.global_start_time = time.time()
        self.min_log_interval = log_interval_in_seconds
        self.last_new_entry_time = -10 ** 10
        self.root_relaxation_entry = None

    def set_global_start_time(self, t):
        self.global_start_time = t

    def add_log_data(self, data, time_within_gurobi, force_new_entry=False):
        try:
            new_entry = LogEntry(globaltime=self._execution_time(), time_within_gurobi=time_within_gurobi, data=data)
            if force_new_entry:  # apply no logic, just add it!
                self.log_entries.append(new_entry)
            elif not self.log_entries:  # always add a new entry when we have none
                self._add_new_log_entry(new_entry)
            elif type(data) != type(self.log_entries[-1].data):
                self._add_new_log_entry(new_entry)
            elif self._is_within_replacement_time_window(time_within_gurobi):
                self._replace_last_log_entry(new_entry)
            else:
                self._add_new_log_entry(new_entry)
        except Exception as e:
            stacktrace = "\nError while adding log entry {} after {:.3f}s, forced={}:\n{}".format(
                data, time_within_gurobi, force_new_entry,
                traceback.format_exc(limit=100)
            )
            for line in stacktrace.split("\n"):
                print(line)
            raise e

    def set_root_relaxation_entry(self, data, time_within_gurobi):
        if self.root_relaxation_entry is not None:
            raise TemporalLogError("Tried to overwrite existing Root relaxation entry {} with {} at time {}".format(
                self.root_relaxation_entry, data, time_within_gurobi
            ))
        self.root_relaxation_entry = LogEntry(globaltime=self._execution_time(), time_within_gurobi=time_within_gurobi, data=data)

    def _execution_time(self):
        current_time = time.time()
        return current_time - self.global_start_time

    def _add_new_log_entry(self, new_entry):
        self.last_new_entry_time = new_entry.time_within_gurobi
        if isinstance(new_entry.data, MIPData):
            if not self.log_entries:
                self.improved_entries.append(new_entry)
            else:
                last_entry = self.log_entries[-1]
                if (isinstance(last_entry.data, LPData) or
                        abs(new_entry.data.objective_value - last_entry.data.objective_value) > 0.0001):
                    self.improved_entries.append(new_entry)
        self.log_entries.append(new_entry)

    def _replace_last_log_entry(self, new_entry):
        last_entry = self.log_entries[-1]
        last_improved_entry = self.improved_entries[-1]

        # replace with updated entry:
        self.log_entries[-1] = new_entry
        # update the improved_entries:
        if isinstance(new_entry.data, MIPData):
            if isinstance(last_entry.data, MIPData):
                # if the last entry was an improvement, overwrite it there too:
                if last_improved_entry == last_entry:
                    self.improved_entries[-1] = new_entry
                # otherwise, check if this one is an improvement & add if necessary:
                elif last_entry.data.objective_value != new_entry.data.objective_value:
                    self.improved_entries.append(new_entry)
            elif isinstance(last_entry.data, LPData):
                if new_entry.data.objective_value != last_improved_entry.data.objective_value:
                    self.improved_entries.append(new_entry)
            else:
                raise TemporalLogError("Last entry {} has invalid data type!".format(last_entry))

    def _is_within_replacement_time_window(self, t):
        last_entry = self.log_entries[-1]
        return ((t - last_entry.time_within_gurobi) < self.min_log_interval and
                (t - self.last_new_entry_time) < self.min_log_interval)

class TemporalLog_Disabled(TemporalLog):
    def __init__(self):
        super(TemporalLog_Disabled, self).__init__(0.0)


    def set_global_start_time(self, t):
        pass

    def add_log_data(self, data, time_within_gurobi, force_new_entry=False):
        pass

    def set_root_relaxation_entry(self, data, time_within_gurobi):
        pass