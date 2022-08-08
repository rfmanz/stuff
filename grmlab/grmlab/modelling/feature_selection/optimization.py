"""
Optimal feature selection problem.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2019.

import os
import re
import threading
import time

import numpy as np

from ..._thirdparty.mipcl.mipshell import BIN
from ..._thirdparty.mipcl.mipshell import INT
from ..._thirdparty.mipcl.mipshell import minimize
from ..._thirdparty.mipcl.mipshell import Problem
from ..._thirdparty.mipcl.mipshell import sum_
from ..._thirdparty.mipcl.mipshell import Var
from ..._thirdparty.mipcl.mipshell import VarVector
from ...core.exceptions import NotSolvedException


_MAX_FEATURES = 25
_MIN_FEATURES = 5
_MICPL_TIME_LIMIT = 100


class FSMILP(Problem):
    """
    Select variables to maximize linear correlation statistic. Current
    formulation handles the following constraints:

        * Lower and upper bound on the number of selected variables.
        * Force selected variables to have equal signs.
        * Pass infeasible pairs of variables, i.e. at most one of the pair
          can be incorporated in the final model.
        * Excluded variables.
        * Fixed variables.
        * Group constraints, i.e, the number of variables of a given group
          present in the final model must be between group size bounds.

    Parameters
    ----------
    nvs : int
        The number of decision variables.

    c : float [nvs]
        The coefficient of each decision variable.

    pmin : int
        Minimum number of selected binary variables.

    pmax : int
        Maximum number of selected binary variables.

    infeas_pairs : list of tuple.
        List of indexes representing infeasible pair sets.

    infeas_features : list
        List of indexes of indexes to be excluded from the optimal set of
        variables due to not satisfying constraints.

    excluded : list
        List of indexes to be excluded from the optimal set of variables.

    fixed : list
        List of indexes to be fixed. These indexes must be included in the
        optimal set of variables.

    groups : list of tuples
        List of group index with minimum and maximum number of variables for
        each group.
    """
    def model(self, nvs, c, pmin, pmax, infeas_pairs, infeas_features,
              excluded, fixed, groups):

        self.c = c

        # decision variables
        # ======================================================================
        self.w = w = VarVector([nvs], "w", BIN)
        self.d = d = Var("d", INT, lb=0, ub=pmax - pmin)

        # objective function
        # ======================================================================
        minimize(sum_(c[j] * w[j] for j in range(nvs)))

        # constraints
        # ======================================================================
        # min/max features
        d + sum_(w[j] for j in range(nvs)) == pmax

        # infeasible pair of features in the model
        for i, j in infeas_pairs:
            w[i] + w[j] <= 1

        # infeasible features in the model
        for i in infeas_features:
            w[i] == 0

        # excluded features
        for i in excluded:
            w[i] == 0

        # fixed features
        for i in fixed:
            w[i] == 1

        # group features (TODO: optimize to reduce fill-in)
        for grp_indexes, n_min, n_max in groups:
            sum_(w[i] for i in grp_indexes) >= n_min
            sum_(w[i] for i in grp_indexes) <= n_max

    def solution(self):
        """"Return boolean numpy array."""
        return np.asarray([self.w[i].val for i in range(len(self.w))],
                          dtype=bool)

    def run(self, timeLimit, silent=False):
        # run optimizer
        self.optimize(timeLimit=timeLimit, silent=silent)
        # remove mp
        self.mp = None


class FeatureSelectionSolver(object):
    """
    FeatureSelectionSolver: solve optimization problem to select features in to
    maximize predictive power subject to business constraints.
    """
    def __init__(self, c, n_min_features, n_max_features, infeas_pairs,
                 infeas_features, excluded, fixed, groups, verbose=False):

        self.c = c
        self.n_min_features = n_min_features
        self.n_max_features = n_max_features
        self.infeas_pairs = infeas_pairs
        self.infeas_features = infeas_features
        self.excluded = excluded
        self.fixed = fixed
        self.groups = groups
        self.verbose = verbose

        # problem input size
        self._nvars = None

        # MIPCL solver options
        self._time_limit = _MICPL_TIME_LIMIT

        # problem status
        self._is_solution = False
        self._is_solution_optimal = False
        self._is_infeasible = False
        self._is_unbounded = False

        # problem statistics
        self._mipcl_problem = None
        self._mipcl_msg = None
        self._mipcl_obj = None
        self._nvs = None
        self._ncs = None
        self._nnz = None
        self._nvs_preprocessing = None
        self._ncs_preprocessing = None
        self._nnz_preprocessing = None
        self._nvs_removed = None
        self._ncs_removed = None
        self._nnz_removed = None
        self._cuts_generated = None
        self._cuts_used = None
        self._branch_and_cut_nodes = None
        self._iters_time = []
        self._iters_obj = []
        self._iters_cuts = []

        # feature selection results
        self._n_selected_features = None
        self._n_infeas_pairs = None
        self._n_infeas_features = None
        self._n_excluded = None
        self._n_fixed = None
        self._n_groups = None
        self._n_groups_variables = None

        # timing statistics
        self._time_total = None
        self._time_solver = None
        self._time_problem_generation = None
        self._time_optimizer_preprocessing = None
        self._time_optimizer = None
        self._time_post_analysis = None

        # auxiliary
        self._mask = None

        # flags
        self._is_solved = False

    def solve(self):
        """Start solving the Feature selection problem."""
        start = time.perf_counter()

        # check FeatureSelectionSolver options
        # ======================================================================

        # objective function class
        if self.n_min_features < 0:
            raise ValueError("min_n_features must be a positive integer.")

        if self.n_max_features < 0:
            raise ValueError("max_n_features must be a positive integer.")

        if self.n_min_features > self.n_max_features:
            raise ValueError("min_n_features must be <= max_n_features.")

        if not isinstance(self.infeas_pairs, (list, np.ndarray)):
            raise TypeError("infeas_pairs must be a list or numpy array.")

        if not isinstance(self.infeas_features, (list, np.ndarray)):
            raise TypeError("infeas_features must be a list or numpy array.")

        if not isinstance(self.excluded, (list, np.ndarray)):
            raise TypeError("excluded must be a list or numpy array.")

        # sizes and target type
        self._nvars = len(self.c)

        # number of constraints
        self._n_infeas_pairs = len(self.infeas_pairs)
        self._n_infeas_features = len(self.infeas_features)
        self._n_excluded = len(self.excluded)
        self._n_fixed = len(self.fixed)
        self._n_groups = len(self.groups)
        self._n_groups_variables = sum(len(g[0]) for g in self.groups)

        # solve MIP optimization problem using MIPCL solver
        self._optimize()

        # parse solver output to compute problem statistics
        self._statistics()

        # update flag
        self._is_solved = True

        self._time_total = time.perf_counter() - start

    def get_support(self):
        """Get the boolean mask indicating which features are selected."""
        if not self._is_solved:
            raise NotSolvedException(self)

        return self._mask

    def summary_statistics(self):
        """
        Generate statistics of the feature selection process and print report.
        """
        if not self._is_solved:
            raise NotSolvedException(self)

        self._stats_report()

    def _optimize(self):
        """Build MIPCL model and solve."""
        start = time.perf_counter()

        self._mipcl_problem = FSMILP("FS_solver")

        # fit optimization problem
        self._mipcl_problem.model(
            self._nvars, self.c, self.n_min_features, self.n_max_features,
            self.infeas_pairs, self.infeas_features, self.excluded,
            self.fixed, self.groups)

        # run solver and catch C++ std output for post-analysis
        self._run_optimizer()

        # post-solve information
        self._is_solution = self._mipcl_problem.is_solution
        self._is_solution_optimal = self._mipcl_problem.is_solutionOptimal
        self._is_infeasible = self._mipcl_problem.is_infeasible
        self._is_unbounded = self._mipcl_problem.is_unbounded

        # retrieve boolean mask with features selected
        self._mask = self._mipcl_problem.solution()
        self._n_selected_features = np.count_nonzero(self._mask)

        # is solution is optimal get value otherwise return NaN, this will
        # be output for infeasible and unbounded problems.
        if self._is_solution_optimal:
            self._mipcl_obj = self._mipcl_problem.getObjVal()
        else:
            self._mipcl_obj = np.NaN

        self._time_solver = time.perf_counter() - start

    def _run_optimizer(self):
        """
        Capture C++ std output from the MIPCL solver. Use mutable objects
        through threads. Jupyter Notebook replaces std I/O by their own
        custom implementations. The following codes circumvents this issue,
        however, last output line is not captured and appears first in
        subsequent runs, this issue need to be taken into account when parsing
        the log file.
        """

        # prepare std output

        # file descriptor unique id (UNIX / Windows) is sys.stdout.fileno(),
        # however, as aforementioned this would not work in Jupyter.
        # By default, fileno = 1 ==> standard output.
        stdout_fileno = 1
        stdout_save = os.dup(stdout_fileno)
        stdout_pipe_read, stdout_pipe_write = os.pipe()

        # copy current stdout
        os.dup2(stdout_pipe_write, stdout_fileno)
        os.close(stdout_pipe_write)

        # prepare list to collect solver messages
        msgs = []

        # trigger thread
        t = threading.Thread(target=self._catpure_solver_output,
                             args=(stdout_pipe_read, msgs, ))

        t.start()

        # run optimizer
        self._mipcl_problem.run(timeLimit=self._time_limit, silent=False)
        # close stdout and collect thread
        os.close(stdout_fileno)
        t.join()

        # clean up the pipe and restore original stdout
        os.close(stdout_pipe_read)
        os.dup2(stdout_save, stdout_fileno)
        os.close(stdout_save)

        # construct output message (small fix is required)
        self._mipcl_msg = ''.join(msgs)

    def _statistics(self):
        """Collect FeatureSelectionSolver and MIPCL statistics."""
        start = time.perf_counter()

        stats = self._parser_solver_output()

        # problem statistics
        self._ncs = stats[0]
        self._nvs = stats[1]
        self._nnz = stats[2]

        if self._is_solution_optimal:
            self._ncs_preprocessing = stats[3]
            self._nvs_preprocessing = stats[4]
            self._nnz_preprocessing = stats[5]
            self._ncs_removed = self._ncs_preprocessing - self._ncs
            self._nvs_removed = self._nvs_preprocessing - self._nvs
            self._nnz_removed = self._nnz_preprocessing - self._nnz

            # objective value, cuts and time per iteration
            self._iters_time = stats[6]
            self._iters_obj = stats[7]
            self._iters_cuts = stats[8]

            # cuts information
            self._cuts_generated = stats[9]
            self._cuts_used = stats[10]
            self._branch_and_cut_nodes = stats[13]

            # timing
            self._time_optimizer_preprocessing = stats[11]
            self._time_optimizer = stats[12]
            self._time_problem_generation = self._time_solver
            self._time_problem_generation -= self._time_optimizer

            # add final iteration
            self._iters_time.append(self._time_optimizer)
            self._iters_obj.append(self._mipcl_obj)
            self._iters_cuts.append(np.NaN)
        else:  # infeasible / unbounded case
            self._mipcl_obj = np.NaN

            self._ncs_preprocessing = np.NaN
            self._nvs_preprocessing = np.NaN
            self._nnz_preprocessing = np.NaN
            self._ncs_removed = np.NaN
            self._nvs_removed = np.NaN
            self._nnz_removed = np.NaN

            self._iters_time = np.NaN
            self._iters_obj = np.NaN
            self._iters_cuts = np.NaN

            self._cuts_generated = 0
            self._cuts_used = 0
            self._branch_and_cut_nodes = 0

            self._time_optimizer_preprocessing = 0
            self._time_optimizer = 0
            self._time_problem_generation = self._time_solver

        self._time_post_analysis = time.perf_counter() - start

    def _parser_solver_output(self):
        """Parser for MIPCL output."""
        lines = self._mipcl_msg.split("\n")

        is_start_pre = False
        is_after_pre = False
        is_optimizing = False
        # is_generating_cuts = False
        is_cut_statistics = False

        ncs_original = None
        nvs_original = None
        nnz_original = None
        ncs_after = None
        nvs_after = None
        nnz_after = None

        iters_time = []
        iters_obj = []
        iters_cuts = []

        cuts_generated = None
        cuts_used = None

        preprocessing_time = None
        solution_time = None
        branch_and_cut_nodes = None

        regex_basic = re.compile("(\d+)")
        regex_decimal = re.compile("(0|([1-9]\d*))(\.\d+)?")

        for line in lines:
            if "Start preprocessing" in line:
                if not is_start_pre:
                    data = regex_basic.findall(line)
                    ncs_original = int(data[0])
                    nvs_original = int(data[1])
                    nnz_original = int(data[4])
                    is_start_pre = True
                else:
                    continue

            if "After preprocessing" in line:
                if not is_after_pre:
                    data = regex_basic.findall(line)
                    ncs_after = int(data[0])
                    nvs_after = int(data[1])
                    nnz_after = int(data[4])
                    is_after_pre = True
                else:
                    continue

            if "Preprocessing Time" in line:
                preprocessing_time = float(regex_decimal.search(line).group(0))

            elif "Optimizing..." in line:
                is_optimizing = True

            elif is_optimizing and not is_cut_statistics:
                data = [f.group(0) for f in regex_decimal.finditer(line)]
                if len(data) == 4:
                    if float(data[1]) == int(float(data[1])):
                        iters_time.append(float(data[0]))
                        iters_obj.append(float(data[3]))
                        iters_cuts.append(np.NaN)
                    else:
                        iters_time.append(float(data[0]))
                        iters_obj.append(float(data[1]))
                        iters_cuts.append(int(data[3]))

            if "Cut statistics" in line:
                is_cut_statistics = True
                is_optimizing = False

            elif is_cut_statistics:
                if "total" in line:
                    cuts_generated, cuts_used = list(
                        map(int, regex_basic.findall(line)))
                    is_cut_statistics = False

            elif "Solution time" in line:
                solution_time = float(regex_decimal.search(line).group(0))

            elif "Branch-and-Cut nodes" in line:
                branch_and_cut_nodes = int(regex_basic.search(line).group(0))

        return [ncs_original, nvs_original, nnz_original, ncs_after, nvs_after,
                nnz_after, iters_time, iters_obj, iters_cuts, cuts_generated,
                cuts_used, preprocessing_time, solution_time,
                branch_and_cut_nodes]

    def _stats_report(self):
        """Report FeatureSelectionSolver statistics."""

        # optimization problem status
        status = "stopped"
        if self._is_solution_optimal:
            status = "optimal"
        elif self._is_infeasible:
            status = "infeasible"
        elif self._is_unbounded:
            status = "unbounded"

        # cuts are not always generated
        if not self._cuts_generated:
            cut_g = 0
            cut_u = 0
            cut_p = 0
        else:
            cut_g = self._cuts_generated
            cut_u = self._cuts_used
            cut_p = cut_u / cut_g

        # % time spent in preprocessing w.r.t optimizer
        if self._time_optimizer:
            rel_pre = self._time_optimizer_preprocessing / self._time_optimizer
        else:
            rel_pre = 0

        # correct timing
        rest = self._time_total
        rest -= self._time_problem_generation
        rest -= self._time_optimizer
        self._time_post_analysis += rest

        report = (
        "\033[94m================================================================================\033[0m\n"
        "\033[1m\033[94m         GRMlab FeatureSelectionSolver 0.1: Feature selection optimizer         \033[0m\n"
        "\033[94m================================================================================\033[0m\n"
        "\n"
        " \033[1mGeneral information                      Configuration options\033[0m\n"
        "   number of variables {:>8}             min features                  {:>3}\n"
        "                                            max features                  {:>3}\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        "\n"
        " \033[1mResults\033[0m\n"
        "   problem class                               {:>30}\n"
        "   selected variables                          {:>30}\n"
        "   infeasible variable pairs                   {:>30}\n"
        "   infeasible variables                        {:>30}\n"
        "   excluded variables                          {:>30}\n"
        "   fixed variables                             {:>30}\n"
        "   group constraints                           {:>30}\n"
        "   group constraints variables                 {:>30}\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        "\n"
        " \033[1mProblem statistics                        Optimizer statistics\033[0m\n"
        "   original problem                          status                {:>10}\n"
        "     variables           {:>6}              objective           {:>.5E}\n"
        "     constraints         {:>6}\n"
        "     nonzeros            {:>6}\n"
        "   after preprocessing                       cutting planes\n"
        "     variables           {:>6} ({:>7})      cuts generated     {:>4}\n"
        "     constraints         {:>6} ({:>7})      cuts used          {:>4} ({:>4.0%})\n"
        "     nonzeros            {:>6} ({:>7})    branch-and-cut nodes {:>4}\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        "\n"
        " \033[1mTiming statistics                         Optimizer options\033[0m\n"
        "   total                 {:>6.3f}              root LP algorithm   {:>10}\n"
        "     model generation    {:>6.3f} ({:>4.0%})       time limit            {:>10}\n"
        "     optimizer           {:>6.3f} ({:>4.0%})       MIP gap                 {:>7.6f}\n"
        "       preprocessing       {:>6.3f} ({:>4.0%})\n"
        "     post-analysis       {:>6.3f} ({:>4.0%})\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        ).format(
            self._nvars, self.n_min_features, self.n_max_features,
            self._mipcl_problem.name, self._n_selected_features,
            self._n_infeas_pairs, self._n_infeas_features,
            self._n_excluded, self._n_fixed, self._n_groups,
            self._n_groups_variables, status, self._nvs, self._mipcl_obj,
            self._ncs,  self._nnz, self._nvs_preprocessing, self._nvs_removed,
            cut_g, self._ncs_preprocessing, self._ncs_removed, cut_u, cut_p,
            self._nnz_preprocessing, self._nnz_removed,
            self._branch_and_cut_nodes, self._time_total, "Dual-Simplex",
            self._time_problem_generation,
            self._time_problem_generation/self._time_total, self._time_limit,
            self._time_optimizer, self._time_optimizer/self._time_total, 0,
            self._time_optimizer_preprocessing, rel_pre,
            self._time_post_analysis,
            self._time_post_analysis/self._time_total)

        print(report)

    @staticmethod
    def _catpure_solver_output(stdout_pipe_read, msgs):
        """Capture solver messages from the pipe."""
        nbytes = 1024
        encoding = "utf-8"
        while True:
            data = os.read(stdout_pipe_read, nbytes)
            if not data:
                break
            msgs.append(data.decode(encoding))
