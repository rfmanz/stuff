"""
Mathematical optimization framework for solving the optimal binning problem.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import os
import re
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.stats import norm
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier

from ..._thirdparty.mipcl.mipshell import BIN
from ..._thirdparty.mipcl.mipshell import INT
from ..._thirdparty.mipcl.mipshell import maximize
from ..._thirdparty.mipcl.mipshell import Problem
from ..._thirdparty.mipcl.mipshell import sum_
from ..._thirdparty.mipcl.mipshell import Var
from ..._thirdparty.mipcl.mipshell import VarVector
from ...core.base import GRMlabBase
from ...core.exceptions import NotFittedException
from .binning import plot
from .binning import table
from .ctree import CTree
from .rtree_categorical import RTreeCategorical
from .util import process_data


def test_proportions(e1, ne1, e2, ne2, zscore):
    n1 = e1 + ne1
    n2 = e2 + ne2
    p1 = e1 / n1
    p2 = e2 / n2
    p = (e1 + e2) / (n1 + n2)

    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    return 1 if abs(z) < zscore else 0


class MIPCLSolver(Problem):
    """
    MIPCL representation of the Optimal Binning optimization problem. A set
    of useful structures for building the problem is included.
    """
    @staticmethod
    def build(nev, ev, max_pvalue):
        """
        Generate auxiliary data: aggregated PD and IV matrices and indicator
        lists for constructing constraints.
        """
        n = len(ev)
        nodes = n * (n + 1) // 2

        t_nev = sum(nev)
        t_ev = sum(ev)
        D = []
        V = []
        E = []
        N = []

        for i in range(1, n+1):
            for j in range(i):
                # probability of default matrix
                d = sum(ev[k] for k in range(j, i)) / sum(
                    nev[k]+ev[k] for k in range(j, i))
                D.append(d)

                # event and nonevent matrix
                if max_pvalue is not None:
                    E.append(sum(ev[k] for k in range(j, i)))
                    N.append(sum(nev[k] for k in range(j, i)))

                # information value matrix
                u = sum(ev[k]/t_ev - nev[k]/t_nev for k in range(j, i))
                v = sum(ev[k]/t_ev for k in range(j, i)) / sum(
                    nev[k]/t_nev for k in range(j, i))
                V.append(u * np.log(v))

        # diagonal
        diag = [(i-1)*(i+2)//2 for i in range(1, n+1)]

        # lower triangular matrix
        low_tri = [i for i in range(nodes) if i not in diag]

        # rows
        rows = [[0]]
        r = list(range(nodes))
        for i in range(1, n):
            rows.append(r[diag[i-1]+1:diag[i]+1])

        # cols
        cols = []
        for j in range(n):
            z = j*(j+3)//2
            k = 0
            col = []
            for i in range(1, n-j+1):
                col.append(z+k)
                k += j+i
            cols.append(col)

        # p-value indexes
        pvalue_indexes = []
        if max_pvalue is not None:
            zscore = norm.ppf(1.0 - max_pvalue / 2)

            for i, row in enumerate(rows[:-1]):
                for r in row:
                    for j in cols[i+1]:
                        if test_proportions(E[r], N[r], E[j], N[j], zscore):
                            pvalue_indexes.append((r, j))

        return D, V, diag, low_tri, rows, cols, t_nev + t_ev, pvalue_indexes

    def model(self, nev, ev, sense, minbs, maxbs, mincs, maxcs, minpd, fixed,
              regularized, reduce_bucket_size_diff, max_pvalue):
        """
        IP/MILP formulation in MIPshell for the optimal binning problem.
        Several formulations are available, including 3 different objective
        functions. The code is not self-explanatory, and it is not aimed to be.

        TODO: challenge ==> preprocessing for convex/concave monotonicity. This
        will allow to tackle problems with a large number of pre-buckets in an
        efficient manner.
        """

        # build problem data and indicator lists
        D, V, diag, low_tri, rows, cols, total, pvalues = self.build(
            nev, ev, max_pvalue)
        self.diag = diag

        # parameters
        # ======================================================================
        n = len(ev)
        nvs = n * (n + 1) // 2

        # regularized parameter (heuristic, it's conservative and works well
        # in practice). Tuning might be required going forward.
        if regularized:
            self.C = C = 2**(-n)

        # auxiliary variables
        self.xx = xx = [1] * n

        # decision variables
        # ======================================================================
        self.x = x = VarVector([nvs], "x", BIN)

        # auxiliary variables for inequality constraints
        if maxcs is not None and mincs is not None:
            self.d = d = Var("d", INT, lb=0, ub=maxcs - mincs)

        # auxiliary variables for 1 change (peak/valley)
        if sense in ("peak", "valley"):
            self.y = y = VarVector([n], "y", BIN)
            self.t = t = Var("t", INT, lb=0)

        # auxiliary variables for reducing pmax-pmin
        if reduce_bucket_size_diff:
            self.pmin = pmin = Var("pmin")
            self.pmax = pmax = Var("pmax")

        # objective function
        # ======================================================================
        if regularized:
            maximize(sum_(V[i] * x[i] for i in diag) +
                     sum_((V[i] - V[i+1]) * x[i] for i in low_tri)
                     - C * sum_(x[i] for i in diag))
        elif reduce_bucket_size_diff:
            maximize(sum_(V[i] * x[i] for i in diag) +
                     sum_((V[i] - V[i+1]) * x[i] for i in low_tri)
                     - (pmax - pmin))
        else:
            maximize(sum_(V[i] * x[i] for i in diag) +
                     sum_((V[i] - V[i+1]) * x[i] for i in low_tri))

        # constraints
        # ======================================================================
        # all sum of columns = 1
        for col in cols:
            sum_(x[i] for i in col) == 1

        # flow continuity
        for i in low_tri:
            x[i+1] - x[i] >= 0

        # monotonicity constraints
        if sense is "ascending":
            for i in range(1, n):
                row = rows[i]
                for p_row in rows[:i]:
                    1 + (D[row[-1]]-1) * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                    ) >= D[p_row[-1]] * x[p_row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                    ) + minpd * (x[row[-1]] + x[p_row[-1]] - 1)

        elif sense is "descending":
            for i in range(1, n):
                row = rows[i]
                for p_row in rows[:i]:
                    D[row[-1]] * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                    ) <= 1 + (D[p_row[-1]] - 1) * x[p_row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                    ) - minpd * (x[row[-1]] + x[p_row[-1]] - 1)

        elif sense is "convex":
            for i in range(2, n):
                row = rows[i]
                for j in range(1, i):
                    p_rowj = rows[j]
                    for k in range(j):
                        p_rowk = rows[k]
                        (D[row[-1]] * x[row[-1]] + sum_(
                        (D[z] - D[z+1]) * x[z] for z in row[:-1])
                        )- 2 * (D[p_rowj[-1]] * x[p_rowj[-1]] + sum_(
                        (D[z] - D[z+1]) * x[z] for z in p_rowj[:-1])
                        ) + (D[p_rowk[-1]] * x[p_rowk[-1]] + sum_(
                        (D[z] - D[z+1]) * x[z] for z in p_rowk[:-1])
                        ) >= (x[row[-1]] + x[p_rowj[-1]] + x[p_rowk[-1]] -3)

        elif sense is "concave":
            for i in range(2, n):
                row = rows[i]
                for j in range(1, i):
                    p_rowj = rows[j]
                    for k in range(j):
                        p_rowk = rows[k]
                        -(D[row[-1]] * x[row[-1]] + sum_(
                        (D[z] - D[z+1]) * x[z] for z in row[:-1])
                        ) + 2 * (D[p_rowj[-1]] * x[p_rowj[-1]] + sum_(
                        (D[z] - D[z+1]) * x[z] for z in p_rowj[:-1])
                        ) - (D[p_rowk[-1]] * x[p_rowk[-1]] + sum_(
                        (D[z] - D[z+1]) * x[z] for z in p_rowk[:-1])
                        ) >= (x[row[-1]] + x[p_rowj[-1]] + x[p_rowk[-1]] -3)

        elif sense in ("peak", "valley"):
            # Big-M := n*n
            for i in range(n):
                t >= i - n * (1 - y[i])
                t <= i + n * y[i]

            if sense is "valley":
                for i in range(1, n):
                    row = rows[i]
                    for z, p_row in enumerate(rows[:i]):
                        n*(y[i] + y[z]) + 1 + (D[row[-1]]-1) * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                        ) >= D[p_row[-1]] * x[p_row[-1]] + sum_(
                            (D[j] - D[j+1]) * x[j] for j in p_row[:-1])

                        D[row[-1]] * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                        ) <= 1 + (D[p_row[-1]] - 1) * x[p_row[-1]] + sum_(
                            (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                        ) + n*(2-y[i] - y[z])
            else: # peak
                for i in range(1, n):
                    row = rows[i]
                    for z, p_row in enumerate(rows[:i]):
                        n*(2-y[i] - y[z]) + 1 + (D[row[-1]]-1) * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                        ) >= D[p_row[-1]] * x[p_row[-1]] + sum_(
                            (D[j] - D[j+1]) * x[j] for j in p_row[:-1])

                        D[row[-1]] * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                        ) <= 1 + (D[p_row[-1]] - 1) * x[p_row[-1]] + sum_(
                            (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                        ) + n*(y[i] + y[z])

        # min/max buckets
        if maxcs is not None and mincs is not None:
            d + sum_(x[i] for i in diag) == maxcs
        elif maxcs is not None:
            sum_(x[i] for i in diag) <= maxcs
        elif mincs is not None:
            sum_(x[i] for i in diag) >= mincs

        # reduce diff between largest and smallest bucket size
        if reduce_bucket_size_diff:
            for i in range(n):
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) <= pmax
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) >= pmin - total / n * (1-x[diag[i]])          

        # max bucket size
        if maxbs < total:
            for i in range(n):
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) <= maxbs * x[diag[i]]

        # min bucket size
        for i in range(n):
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) >= minbs * x[diag[i]]

        # max p-value
        for i, j in pvalues:
            x[i] + x[j] <= 1

        # preprocessing
        # ======================================================================
        # search infeasible solutions and fix split to 0
        if sense is "ascending":
            for i in range(1, n-1):
                row = rows[i]
                if D[row[i-1]] - D[row[i]] > 0:
                    x[diag[i-1]] == 0
                    xx[i-1] = 0
                    
                    # important, otherwise infeasible (max bucket size const)
                    if maxbs == total or not reduce_bucket_size_diff:
                        for j in range(1, n-i-1):
                            row = rows[j+i]
                            if D[row[i-1]] - D[diag[i+j]] > 0:
                                x[diag[i+j-1]] == 0
                                xx[i+j-1] = 0

            # search infeasible solutions when minpd > 0
            if minpd:
                for i in range(1, n-1):
                    if (max(D[diag[j-1]] * xx[j-1] for j in range(i+1, n))
                        - D[diag[i-1]]) < minpd:
                        x[diag[i-1]] == 0
                        xx[i-1] = 0

        elif sense is "descending":
            for i in range(1, n-1):
                row = rows[i]
                if D[row[i-1]] - D[row[i]] < 0:
                    x[diag[i-1]] == 0
                    xx[i-1] = 0
                    
                    # important, otherwise infeasible (max bucket size const)
                    if maxbs == total or not reduce_bucket_size_diff:
                        for j in range(1, n-i-1):
                            row = rows[j+i]
                            if D[row[i-1]] - D[diag[i+j]] < 0:
                                x[diag[i+j-1]] == 0
                                xx[i+j-1] = 0

            # search infeasible solutions when minpd > 0
            if minpd:
                for i in range(1, n-1):
                    if (D[diag[i-1]] - min(D[diag[j-1]] * xx[j-1] for j in 
                        range(i+1, n))) < minpd:
                        x[diag[i-1]] == 0
                        xx[i-1] = 0

        # fixed buckets
        if fixed:
            for i in fixed:
                x[diag[i]] == 1

    def model_2(self, nev, ev, sense, minbs, maxbs, mincs, maxcs, minpd, fixed,
                regularized, reduce_bucket_size_diff, t, max_pvalue):
        """
        IP/MILP formulation in MIPshell for the optimal binning problem.
        Efficient formulation for peak/valley monotonicity. Several
        formulations are available, including 3 different objective functions.

        Effective preprocessing for peak/valley monotonicity based on
        ascending/descending algorithm.
        """

        # build problem data and indicator lists
        D, V, diag, low_tri, rows, cols, total, pvalues = self.build(
            nev, ev, max_pvalue)
        self.diag = diag

        # parameters
        # ======================================================================
        n = len(ev)
        nvs = n * (n + 1) // 2

        # regularized parameter (heuristic, it's conservative and works well
        # in practice). Tuning might be required going forward.
        if regularized:
            self.C = C = 2**(-n)

        # auxiliary variables
        self.xx = xx = [1] * n

        # decision variables
        # ======================================================================
        self.x = x = VarVector([nvs], "x", BIN)
        
        # auxiliary variables for inequality constraints
        if maxcs is not None and mincs is not None:
            self.d = d = Var("d", INT, lb=0, ub=maxcs - mincs)

        # auxiliary variables for reducing pmax-pmin
        if reduce_bucket_size_diff:
            self.pmin = pmin = Var("pmin")
            self.pmax = pmax = Var("pmax")

        # objective function
        # ======================================================================
        if regularized:
            maximize(sum_(V[i] * x[i] for i in diag) + 
                     sum_((V[i] - V[i+1]) * x[i] for i in low_tri)
                     - C * sum_(x[i] for i in diag))
        elif reduce_bucket_size_diff:
            maximize(sum_(V[i] * x[i] for i in diag) + 
                     sum_((V[i] - V[i+1]) * x[i] for i in low_tri) 
                     - (pmax - pmin))
        else:
            maximize(sum_(V[i] * x[i] for i in diag) + 
                     sum_((V[i] - V[i+1]) * x[i] for i in low_tri))
                
        # constraints
        # ======================================================================
        # all sum of columns = 1
        for col in cols:
            sum_(x[i] for i in col) == 1
        
        # flow continuity
        for i in low_tri:
            x[i+1] - x[i] >= 0
        
        # monotonicity constraints
        if sense is "peak":
            for i in range(1, t):
                row = rows[i]
                for p_row in rows[:i]:
                    1 + (D[row[-1]]-1) * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                    ) >= D[p_row[-1]] * x[p_row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                    ) + minpd * (x[row[-1]] + x[p_row[-1]] - 1)         

            for i in range(t, n):
                row = rows[i]
                for p_row in rows[t:i]:
                    D[row[-1]] * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                    ) <= 1 + (D[p_row[-1]] - 1) * x[p_row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                    ) - minpd * (x[row[-1]] + x[p_row[-1]] - 1)

        if sense is "valley":
            for i in range(1, t):
                row = rows[i]
                for p_row in rows[:i]:
                    D[row[-1]] * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                    ) <= 1 + (D[p_row[-1]] - 1) * x[p_row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                    ) - minpd * (x[row[-1]] + x[p_row[-1]] - 1)

            for i in range(t, n):
                row = rows[i]
                for p_row in rows[t:i]:
                    1 + (D[row[-1]]-1) * x[row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in row[:-1]
                    ) >= D[p_row[-1]] * x[p_row[-1]] + sum_(
                        (D[j] - D[j+1]) * x[j] for j in p_row[:-1]
                    ) + minpd * (x[row[-1]] + x[p_row[-1]] - 1) 

        # min/max buckets
        if maxcs is not None and mincs is not None:
            d + sum_(x[i] for i in diag) == maxcs
        elif maxcs is not None:
            sum_(x[i] for i in diag) <= maxcs
        elif mincs is not None:
            sum_(x[i] for i in diag) >= mincs

        # reduce difference between largest and smallest bucket size
        if reduce_bucket_size_diff:
            for i in range(n):
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) <= pmax
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) >= pmin - total / n * (1-x[diag[i]])          

        # max bucket size
        if maxbs < total:
            for i in range(n):
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) <= maxbs * x[diag[i]]

        # min bucket size
        for i in range(n):
                sum_((nev[k] + ev[k]) * x[j] for k, j in enumerate(rows[i])
                    ) >= minbs * x[diag[i]]

        # max p-value
        for i, j in pvalues:
            x[i] + x[j] <= 1

        # preprocessing
        # ======================================================================
        # search infeasible solutions and fix split to 0
        if sense is "peak":
            for i in range(1, t-1):
                row = rows[i]
                if D[row[i-1]] - D[row[i]] > 0:
                    x[diag[i-1]] == 0
                    xx[i-1] = 0
                    
                    # check, otherwise infeasible (max bucket size constraint)
                    if maxbs == total or not reduce_bucket_size_diff:
                        for j in range(1, t-i-1):
                            row = rows[j+i]
                            if D[row[i-1]] - D[diag[i+j]] > 0:
                                x[diag[i+j-1]] == 0
                                xx[i+j-1] = 0

            for i in range(t+1, n-1):
                row = rows[i]
                if D[row[i-1]] - D[row[i]] < 0:
                    x[diag[i-1]] == 0
                    xx[i-1] = 0
                    
                    # check, otherwise infeasible (max bucket size constraint)
                    if maxbs == total or not reduce_bucket_size_diff:
                        for j in range(t+1, n-i-1):
                            row = rows[j+i]
                            if D[row[i-1]] - D[diag[i+j]] < 0:
                                x[diag[i+j-1]] == 0
                                xx[i+j-1] = 0                                   

        else: # valley
            for i in range(1, t-1):
                row = rows[i]
                if D[row[i-1]] - D[row[i]] < 0:
                    x[diag[i-1]] == 0
                    xx[i-1] = 0
                    
                    # check, otherwise infeasible (max bucket size constraint)
                    if maxbs == total or not reduce_bucket_size_diff:
                        for j in range(1, t-i-1):
                            row = rows[j+i]
                            if D[row[i-1]] - D[diag[i+j]] < 0:
                                x[diag[i+j-1]] == 0
                                xx[i+j-1] = 0

            for i in range(t+1, n-1):
                row = rows[i]
                if D[row[i-1]] - D[row[i]] > 0:
                    x[diag[i-1]] == 0
                    xx[i-1] = 0
                    
                    # check, otherwise infeasible (max bucket size constraint)
                    if maxbs == total or not reduce_bucket_size_diff:
                        for j in range(t+1, n-i-1):
                            row = rows[j+i]
                            if D[row[i-1]] - D[diag[i+j]] > 0:
                                x[diag[i+j-1]] == 0
                                xx[i+j-1] = 0                       

        # fixed buckets
        if fixed:
            for i in fixed:
                x[diag[i]] == 1

    def solution(self):
        return [int(self.x[i].val) for i in self.diag]

    def infeasible_buckets(self):
        return len(self.xx) - sum(self.xx)

    def run(self, timeLimit, silent=False):
        # run optimizer
        self.optimize(timeLimit=timeLimit, silent=silent)
        # remove mp
        self.mp = None


class OptBin(GRMlabBase):
    """
    OptBin algorithm to perform optimal feature binning.

    OptBin solves a mixed-integer linear programming (IP/MILP) optimization
    problem via a rigorous approach.

    Parameters
    ----------
    name : str
        The variable name.

    dtype : str (default="numerical")
        The variable data type. Four dtypes supported: "categorical", "nominal",
        "numerical" and "ordinal".

    prebinning_algorithm : str (default="rtree")
        The prebinning algorithm. Algorithm supported are: "ctree" and "rtree".
        Option "ctree" uses ``grmlab.data_processing.feature_binning.CTree``
        algorithm, whereas "rtree" uses CART algorithm implementation
        ``sklearn.tree.DecisionTreeClassifier`` and
        ``grmlab.data_processing.feature_binning.RTreeCategorical``. "rtree" is
        the only one that supports weights.

    rtree_max_leaf_nodes : int (default=20)
        The maximum number of leaf nodes.

    ctree_min_criterion: float (default=0.95)
        The value of the test statistic or 1 - (alpha or significance level)
        that must be exceeded in order to add a split.

    ctree_max_candidates : int (default=64)
        The maximum number of split points to perform an exhaustive search.

    ctree_dynamic_split_method : str (default="k-tile")
        The method to generate dynamic split points. Supported methods are
        “gaussian” for the Gaussian approximation, “k-tile” for the quantile
        approach and “entropy+k-tile” for a heuristic using class entropy. The
        "entropy+k-tile" method is only applicable when target is binary,
        otherwise, method "k-tile" is used instead.

    prebinning_others_group : boolean (default=True)
        Whether to create an extra group with those values (categories) do not
        sufficiently representative. This option is available for dtypes
        "categorical" and "nominal".

    prebinning_others_threshold : float (default=0.01)
        Merge categories which percentage of cases is below threshold to create
        an extra group. This option is available for dtypes "categorical" and
        "nominal".

    min_buckets : int or None (default=None)
        The minimum number of optimal buckets. If None then unlimited number of
        buckets.

    max_buckets : int or None (default=None)
        The maximum number of optimal buckets. If None then unlimited number of
        buckets.

    min_bucket_size : float (default=0.05)
        The minimum number of records per bucket. Percentage of total records.

    max_bucket_size : float (default=1.0)
        The maximum number of records per bucket. Percentage of total records.

    monotonicity_sense : str (default="auto")
        The event rate monotonicity sense. Supported options are: "auto",
        "ascending", "descending", "concave", "convex", "peak" and "valley".
        Option "auto" estimates the monotonicity sense maximizing discriminant
        power.

    monotonicity_force : boolean (default=False)
        Force ascending or descending monotonicity when
        ``monotonicity_sense="auto"``.

    min_er_diff : float (default=0.0)
        The minimum difference between event rates of consecutive buckets.

    regularization : boolean (default=False)
        Try to reduce the number of optimal buckets using a regularization
        factor.

    reduce_bucket_size_diff : boolean (default=False)
        Try to reduce standard deviation among size of optimal buckets. This
        aims to produce solutions with more homogeneous bucket sizes, thus
        avoiding cases where only a few buckets are dominant.

    pvalue_method : str (default="Chi2")
        The statistical test to determine whether two consecutive buckets are
        significantly different. Two methods are supported: "Chi2" uses
        ``scipy.stats.chi2_contingency`` and "Fisher" using
        ``scipy.stats.fisher_exact``.

    max_pvalue : float or None (default=None):
        The maximum allowed p-value between consecutive buckets.

    user_splits : list or None (default=None)
        List of prebinning split points provided by a user.

    user_idx_forced_splits : list or None (default=None)
        Indexes of split points to be fixed. The optimal solution must include
        those splits.

    special_values : list or None (default=None)
        List of special values to be considered.

    special_handler_policy : str (default="join")
        Method to handle special values. Options are "join", "separate" and
        "binning". Option "join" creates an extra bucket containing all special
        values. Option "separate" creates an extra bucket for each special
        value. Option "binning" performs feature binning of special values using
        ``grmlab.data_processing.feature_binning.CTree`` in order to split
        special values if these are significantly different.

    special_woe_policy : str (default=empirical)
        Weight-of-Evidence (WoE) value to be assigned to special values buckets.
        Options supported are: "empirical", "worst", "zero". Option "empirical"
        assign the actual WoE value. Option "worst" assigns the WoE value
        corresponding to the bucket with the highest event rate. Finally, option
        "zero" assigns value 0.

    missing_woe_policy : str (default="empirical")
        Weight-of-Evidence (WoE) value to be assigned to missing values bucket.
        Options supported are: "empirical", "worst", "zero". Option "empirical"
        assign the actual WoE value. Option "worst" assigns the WoE value
        corresponding to the bucket with the highest event rate. Finally, option
        "zero" assigns value 0.

    verbose: int or boolean (default=False)
        Controls verbosity of output.

    See also
    --------
    CTree

    Example
    -------
    >>> from grmlab.data_processing.feature_binning import OptBin
    >>> from sklearn.datasets import make_classification
    >>> X, y =  make_classification(n_samples=1000000, n_features=2, n_informative=2, n_redundant=0)
    >>> x = X[:, 1]
    >>> opt = OptBin(max_buckets=10)
    >>> opt.fit(x, y)
    >>> opt.splits_optimal
    array([-1.57772285, -1.12774384, -0.79361489, -0.36901367,  0.12206733,
        0.46386676,  0.7109344 ,  0.95995894,  1.81078154])
    """
    def __init__(self, name="", dtype="numerical", prebinning_algorithm="rtree",
        rtree_max_leaf_nodes=20, ctree_min_criterion=0.95,
        ctree_max_candidates=64, ctree_dynamic_split_method="k-tile",
        prebinning_others_group=True, prebinning_others_threshold=0.01,
        min_buckets=None, max_buckets=None, min_bucket_size=0.05,
        max_bucket_size=1.0, monotonicity_sense="auto", monotonicity_force=False,
        min_er_diff=0.0, regularization=False, reduce_bucket_size_diff=False,
        pvalue_method="Chi2", max_pvalue=0.05, user_splits=None,
        user_idx_forced_splits=None, special_values=None,
        special_handler_policy="join", special_woe_policy="empirical",
        missing_woe_policy="empirical", verbose=False):

        # main input data
        self.name = name
        self.dtype = dtype

        # pre-binning options (activated)
        self._is_prebinning = True
        self.prebinning_algorithm = prebinning_algorithm

        # rtree parameters
        self.rtree_max_leaf_nodes =  rtree_max_leaf_nodes

        # ctree options
        self._ctree_min_criterion = ctree_min_criterion
        self.ctree_max_candidates = ctree_max_candidates
        self.ctree_dynamic_split_method = ctree_dynamic_split_method
        self.prebinning_others_group = prebinning_others_group
        self.prebinning_others_threshold = prebinning_others_threshold

        # general optbin parameters
        self.monotonicity_user = monotonicity_sense
        self.monotonicity_sense = monotonicity_sense
        self.monotonicity_force = monotonicity_force
        self.min_buckets = min_buckets
        self.max_buckets = max_buckets
        self.min_bucket_size = min_bucket_size
        self.max_bucket_size = max_bucket_size
        self.min_er_diff = min_er_diff
        self.pvalue_method = pvalue_method
        self.max_pvalue = max_pvalue
        self.regularization = regularization
        self.reduce_bucket_size_diff = reduce_bucket_size_diff

        # user-defined splits
        self.user_splits_provided = False
        self.user_splits = [] if user_splits is None else user_splits
        if user_idx_forced_splits is None:
            self.user_idx_forced_splits = []
        else:
            self.user_idx_forced_splits = user_idx_forced_splits

        # special values options
        self.special_values = [] if special_values is None else special_values
        self.special_handler_policy = special_handler_policy
        self.special_woe_policy = special_woe_policy

        # missing values options
        self.missing_woe_policy = missing_woe_policy

        # others
        self.verbose = verbose

        # main dataframe/array characteristics
        self._n_samples= None

        # peak/valley extra parameter
        self._trend_change = None

        # MIPCL solver options
        self._time_limit = 60

        # problem status
        self._is_solution = False
        self._is_solution_optimal = False
        self._is_infeasible = False
        self._is_unbounded = False

        # problem statistics
        self._mipcl_problem = None
        self._mipcl_msg = None
        self._mipcl_obj = None
        self._n_prebuckets = None
        self._n_optimal_buckets = None
        self._infeasible_buckets = None
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

        # optimal-binning results
        self._binning_table_optimal = None
        self._iv_optimal = None
        self._max_pvalue = None
        self._largest_bucket_perc = None
        self._smallest_bucket_perc = None
        self._largest_bucket_size = None
        self._smallest_bucket_size = None
        self._diff_largest_smallest_bucket_size = None
        self._std_bucket_size = None
        self._group_special = None
        self._group_missing = None
        self._group_others = None

        # pre-binning results
        self._iv_prebinning = None
        self._prebinning_trend_changes = None

        # timing statistics
        self._time_total = None
        self._time_prebinning = None
        self._time_problem_data = None
        self._time_solver = None
        self._time_problem_generation = None
        self._time_optimizer_preprocessing = None
        self._time_optimizer = None
        self._time_post_analysis = None

        # cutpoints / splits variables
        self._splits_prebinning = []
        self._splits_optimal = []
        self._splits_specials = []
        self._splits_others = []

        # flags
        self._is_fitted = False

    def fit(self, x, y, sample_weight=None, check_input=True):
        """
        Build optimal binning from the training set (x, y).

        Parameters
        ----------
        x : array-like, shape = [n_samples]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        sample_weight : array-like, shape = [n_samples] (default=None)
            Individual weights for each sample. "rtree" is the only method
            that supports weights.

        check_input : boolean (default=True)
            Option to perform several input checking.

        Returns
        -------
        self : object
        """
        if check_input:
            if not isinstance(x, np.ndarray):
                raise TypeError("x must be a numpy array.")

            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array.")

            if not x.size:
                raise ValueError("x cannot be empty.")

            if not y.size:
                raise ValueError("y cannot be empty.")

            if len(x) != len(y):
                raise ValueError("x and y must have the same length.")

            if sample_weight is not None:
                if not isinstance(sample_weight, np.ndarray):
                    raise TypeError("sample_weight must be a numpy array.")

                if len(x) != len(sample_weight):
                    raise ValueError("x and sample_weight must have the same"
                                     "length.")


        # variable dtype
        if self.dtype not in ("numerical", "ordinal", "nominal",
            "categorical"):
            raise ValueError("dtype not supported.")

        # pre-binning algorithm
        if self.prebinning_algorithm not in ("rtree", "ctree"):
            raise ValueError("prebinning_algorithm not supported.")

        # pre-binning or user-defined splits
        if len(self.user_splits):
            if self._is_prebinning and self.verbose:
                print("pre-binning algorithm is disable.")

            self.user_splits_provided = True
            self._is_prebinning = False

        # monotonicity
        if self.monotonicity_sense not in ("auto", "ascending", "descending", 
            "concave", "convex", "peak", "valley"):
            raise ValueError("monotonicity sense {} not supported. "
                   "Available options: auto, ascending, descending"
                   " concave, convex, valley and peak.".format(
                    self.monotonicity_sense))

        # max/min buckets
        if self.min_buckets is not None and self.max_buckets is not None:
            if self.min_buckets > self.max_buckets:
                raise ValueError("min_buckets must be <= max_buckets.")

        if self.min_buckets is not None and self.min_buckets < 0:
            raise ValueError("min_buckets must be > 0.")

        if self.max_buckets is not None and self.max_buckets < 0:
            raise ValueError("max_buckets must be > 0.")

        # max/min bucket size
        if self.min_bucket_size > self.max_bucket_size:
            raise ValueError("min_bucket_size must be <= max_bucket_size.")

        if self.min_bucket_size < 0 or self.min_bucket_size > 1.0:
            raise ValueError("min_bucket_size must be in (0, 1).")

        if self.max_bucket_size < 0 or self.max_bucket_size > 1.0:
            raise ValueError("max_bucket_size must be in (0, 1).")

        # min PD difference among buckets
        if self.min_er_diff < 0:
            raise ValueError("min_er_diff must be >= 0.")

        # max leaf nodes is limited depending on the monotonicity sense. for 
        # ascending/descending no limitation exist, although more than
        # 100 buckets is a difficult problem to solve.
        if (self.monotonicity_sense in ("concave", "convex")
            and self.rtree_max_leaf_nodes > 20):
            raise ValueError("maximum number of leaf nodes must be <= 20 when "
                    " monotonicity sense is set to 'convex' or 'concave'.")

        # regularization options
        if not isinstance(self.regularization, bool):
            raise ValueError("regularization must be a boolean.")

        if not isinstance(self.reduce_bucket_size_diff, bool):
            raise ValueError("reduce_bucket_size_diff must be a boolean.")

        # p-value method
        if self.pvalue_method not in ("Chi2", "Fisher"):
            raise ValueError("pvalue_method not supported.")

        # max p-value
        if self.max_pvalue is not None:
            if self.max_pvalue < 0.0 or self.max_pvalue > 1.0:
                raise ValueError("max_pvalue must be a float in [0, 1].")

        # ctree_min_criterion
        if self._ctree_min_criterion < 0.5 or self._ctree_min_criterion > 1.0:
            raise ValueError("ctree_min_criterion must be a float in [0.5, 1.0).")

        # ctree_max_candidates
        if self.ctree_max_candidates < 2:
            raise ValueError("ctree_max_candidates must be >= 2.")

        # ctree_dynamic_split_method
        if self.ctree_dynamic_split_method not in ("gaussian", "k-tile", 
            "entropy+k-tile"):
            raise ValueError("ctree_dynamic_split_method not supported.")

        # special values
        if not isinstance(self.special_values, (list, np.ndarray)):
            raise TypeError("special_values must be a list of numpy array.")

        # special_handler_policy
        if self.special_handler_policy not in ("join", "separate", "binning"):
            raise ValueError("special_handler_policy option not supported.")

        # special_woe_policy
        if self.special_woe_policy not in ("empirical", "worst", "zero"):
            raise ValueError("special_woe_policy option not supported.")

        # missing_woe_policy
        if self.missing_woe_policy not in ("empirical", "worst", "zero"):
            raise ValueError("missing_woe_policy option not supported.")

        # fit optbin
        return self._fit(x, y, sample_weight)

    def _fit(self, x, y, sample_weight=None):
        """Start solving the Optimal Binning problem."""
        start = time.perf_counter()
        
        x = x.copy()
        y = y.copy()
        if sample_weight is not None:
            sample_weight = sample_weight.copy()

        self._n_samples= len(x)

        # if user-defined splits are provided do not perform pre-binning, 
        # otherwise run pre-binning solving classification decision tree
        if self.user_splits_provided:
            self._splits_prebinning = np.asarray(self.user_splits)
            self._group_others = 0
            self._time_prebinning = 0
        else:
            self._is_prebinning = True
            self._prebinning(x, y, sample_weight)

        # solve IP optimization problem using MIPCL solver
        self._optimize(x, y, sample_weight)

        # parse solver output to compute problem statistics
        self._statistics(x, y, sample_weight)

        self._time_total = time.perf_counter() - start

        # update flag
        self._is_fitted = True

        return self

    def stats(self):
        """OptBin solver and timing statistics."""
        if not self._is_fitted:
            raise NotFittedException(self)
        
        self._stats_report()

    def transform(self, x):
        """
        Apply Woe transformation to array x.

        Parameters
        ----------
        x : array-like, shape = [n_samples]
            The training input samples.
        """
        if not self._is_fitted:
            raise NotFittedException(self)

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array.")

        n = len(x)

        if self.dtype == "categorical":
            idx_special = np.isin(x, self.special_values)
            idx_nan = pd.isnull(x)
            idx_clean = (~idx_nan) & (~idx_special)
            x[idx_clean] = x[idx_clean].astype(str)

        # indexes with special values
        idx_special = np.isin(x, self.special_values)

        # optimal splits
        n_splits = len(self.splits_optimal)

        # woe from binning table
        woe = self._binning_table_optimal.WoE.values[:-1]

        # initialize woe and groups arrays
        t_woe = np.zeros(n)

        if not n_splits:
            t_woe = np.ones(n) * woe[0]
            idx_nan = pd.isnull(x)
            n_splits = 1

        elif self.dtype in ("categorical", "nominal"):
            # categorical and nominal variables return groups as numpy.ndarray
            for _idx, split in enumerate(self.splits_optimal):
                # mask = np.isin(x, split)
                mask = pd.Series(x).isin(split).values
                t_woe[mask] = woe[_idx]
            # indexes with NaN in x
            idx_nan = pd.isnull(x)

        else:
            # numerical and ordinal variables return extra group (> last split)
            splits = self.splits_optimal[::-1]
            mask = (x > splits[-1])
            t_woe[mask] = woe[n_splits]
            for _idx, split in enumerate(splits):
                mask = (x <= split)
                t_woe[mask] = woe[n_splits - (_idx + 1)]
            # indexes with NaN in x
            idx_nan = pd.isnull(x)
            # account for > group
            n_splits += 1

        # special values
        if self._splits_specials:
            for _idx, split in enumerate(self._splits_specials):
                if isinstance(split, np.ndarray):
                    mask = np.isin(x, split)
                else:
                    mask = (x == split)
                t_woe[mask] = woe[n_splits]
                n_splits += 1
        else:
            t_woe[idx_special] = woe[n_splits]
            n_splits += 1

        # missing values
        t_woe[idx_nan] = woe[n_splits]

        return t_woe

    def binning_table(self):
        """
        Return binning table with optimal buckets / split points.

        Returns
        -------
        binning_table : pandas.DataFrame
        """
        if not self._is_fitted:
            raise NotFittedException(self)
        
        return self._binning_table_optimal

    def plot_binning_table(self, plot_type="pd", plot_bar_type="event"):
        """
        Plot showing how the set of pre-buckets is merged in order to satisfy
        all constraints.

        Parameters
        ----------
        plot_type : str (default="pd")
            The measure to show in y-axis. Options are: "pd" and "woe".

        plot_bar_type : str (default="event")
            The count value to show in barplot. Options are: "all", "event"
            and "nonevent".
        """
        if not self._is_fitted:
            raise NotFittedException(self)
        
        if plot_type not in ("woe", "pd"):
            raise ValueError("plot type not supported.")

        return plot(df=self.binning_table(), splits=self._splits_optimal, 
            plot_type=plot_type, plot_bar_type=plot_bar_type,
            others_group=self._group_others)

    def plot_optimizer_progress(self):
        """
        Plot showing how the solver proceeds to the optimal solution. It also
        includes the number of generated cuts at each iteration.
        """
        if not self._is_fitted:
            raise NotFittedException(self)

        return self._plot_solver_progress(self._iters_time, 
            self._iters_obj, self._iters_cuts)

    def _prebinning(self, x, y, sample_weight=None):
        """
        Run pre-binning phase to generate set of pre-buckets to be merged in
        the optimization phase.
        """
        start = time.perf_counter()

        # minimum number of samples required to be a leaf nodes: is computed
        # from min_bucket_size
        min_samples_leaf = int(self.min_bucket_size * self._n_samples)

        # remove NaN and special values from data
        x, y, sample_weight = process_data(x, y, sample_weight,
                                           self.special_values)

        if self.prebinning_algorithm == "rtree":
            if self.dtype in ("nominal", "categorical"):
                rtree = RTreeCategorical(
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=self.rtree_max_leaf_nodes, presort=True)

                rtree.fit(x, y, sample_weight=sample_weight)

                if rtree.others_group:
                    self._group_others = 1
                    self._splits_prebinning = rtree.splits[:-1]
                    self._splits_others = rtree.splits[-1]

                    if self.verbose:
                        print("pre-binning: group others was generated " 
                            "including {} distinct values.".format(
                                len(self._splits_others)))
                else:
                    self._group_others = 0
                    self._splits_prebinning = rtree.splits
            else:
                # build and fit decision tree classifier (CART)
                rtree = DecisionTreeClassifier(
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=self.rtree_max_leaf_nodes, presort=True)

                rtree.fit(x.reshape(-1, 1), y, sample_weight=sample_weight)

                # retrieve splits
                splits = np.unique(rtree.tree_.threshold)
                self._splits_prebinning = splits[splits != _tree.TREE_UNDEFINED]

                # others group
                self._group_others = 0

        else:
            # build and fit GRMlab CTree
            ctree = CTree(dtype=self.dtype,
                min_samples_leaf=min_samples_leaf, 
                min_criterion=self._ctree_min_criterion,
                max_candidates=self.ctree_max_candidates,
                dynamic_split_method=self.ctree_dynamic_split_method,
                others_group=self.prebinning_others_group,
                others_threshold=self.prebinning_others_threshold,
                verbose=self.verbose)

            ctree.fit(x, y)

            # retrieve splits
            if self.dtype in ("nominal", "categorical"):
                if ctree.others_group:
                    self._group_others = 1
                    self._splits_prebinning = ctree.splits[:-1]
                    self._splits_others = ctree.splits[-1]

                    if self.verbose:
                        print("pre-binning: group others was generated " 
                            "including {} distinct values.".format(
                                len(self._splits_others)))
                else:
                    self._group_others = 0
                    self._splits_prebinning = ctree.splits
            else:
                self._group_others = 0
                self._splits_prebinning = ctree.splits

        self._time_prebinning = time.perf_counter() - start

    def _pre_optimize_data(self, x, y, sample_weight=None):
        """Compute data to pass to the solver."""
        start = time.perf_counter()

        # Note: handle pre-buckets with 0 nonevent or 0 event
        nonevent, event, default = self._check_prebinning(x, y, sample_weight)

        # extra parameters for the optimization problem
        total = sum(nonevent) + sum(event)
        min_bucket_size = int(self.min_bucket_size * total)
        max_bucket_size = int(self.max_bucket_size * total)

        # monotonicity sense: compute decision variables and apply decision
        # tree classifier to determine the sense maximizing IV.
        if self._n_prebuckets == 0:
            if self.verbose:
                print("pre-binning analysis: it is not possible to create any",
                    "significant split. A unique group is returned.")
                print("pre-binning analysis: all splits were removed.")
            self.monotonicity_sense = "undefined"

        elif self._n_prebuckets == 1:
            if self.verbose:
                print("pre-binning analysis: it is not possible to create any", 
                    "significant split. A unique group is returned.")
                print("pre-binning analysis: monotonicity is set to ascending.")
            self.monotonicity_sense = "ascending"

        elif self.monotonicity_sense == "auto":
            if self.dtype in ("nominal", "categorical"):
                self.monotonicity_sense = "ascending"
            else:
                params = self._monotonicity_parameters(nonevent, event, default)
                self.monotonicity_sense = self._monotonicity_decision(*params)

            if self.verbose:
                print("pre-binning analysis: monotonicity decision engine",
                    "detects {} sense.".format(self.monotonicity_sense))

        elif self.monotonicity_sense in ("peak", "valley"):
            self._compute_trend_change(default)

        # compute trend changes in pre-binning solution
        self._prebinning_trend_changes = self._trend_changes(default)

        self._time_problem_data = time.perf_counter() - start

        return nonevent, event, min_bucket_size, max_bucket_size

    def _check_prebinning(self, x, y, sample_weight=None):
        """
        Detect those buckets with number of event or number of nonevent equal 
        to zero. Two procedures are considered:

        1) Remove the splits corresponding to those cases and reconstruct the
            IV table.

        2) Use an alternative metric such as Hellinger or Taneja's divergence
            to maximize and add two constraints to guarantee that an optimal
            buckets has at least one event and one nonevent.

        Currently only the first approach is implemented.
        """
        # construct binning table
        ivt, _ = table(x, y, self._splits_prebinning, sample_weight,
            self.special_handler_policy, self.special_values, 
            self.special_woe_policy, self.missing_woe_policy)

        # extract nonevent and event records by split
        if self.dtype in ("nominal", "categorical"):
            n_buckets = len(self._splits_prebinning)
        else:   
            n_buckets = len(self._splits_prebinning) + 1

        nev = list(ivt.iloc[:, 2].values[:n_buckets])
        ev = list(ivt.iloc[:, 3].values[:n_buckets])

        # new splits satisfying nonevent and event criteria
        if self.dtype in ("nominal", "categorical"):
            idx = (np.isin(nev, [0,1]) | np.isin(ev, [0,1]))
        else:
            idx = (np.isin(nev[:-1], [0,1]) | np.isin(ev[:-1], [0,1]))
        s = self._splits_prebinning[~idx]

        removed_splits = self._splits_prebinning[idx]

        if len(self._splits_prebinning) == len(s):          
            self._n_prebuckets = n_buckets
            nonevent = nev
            event = ev
            default = ivt.iloc[:, 4].values[:n_buckets]

        else: # some splits were removed
            n_removed_splits = len(self._splits_prebinning) - len(s)
            if self.verbose:
                if n_removed_splits > 1:
                    print("pre-binning analysis: {} splits were removed."
                        .format(n_removed_splits))
                else:
                    print("pre-binning analysis: 1 split was removed.")

                for rs in removed_splits:
                    print("pre-binning analysis: split {} was removed."
                        .format(rs))
                print("pre-binning analysis: re-constructed binning table.")

            # update splits and generate new binning table
            self._splits_prebinning = s

            ivt, _ = table(x, y, self._splits_prebinning, sample_weight,
                self.special_handler_policy, self.special_values, 
                self.special_woe_policy, self.missing_woe_policy)

            # extract nonevent and event records by split
            if self.dtype in ("nominal", "categorical"):
                n_buckets = len(self._splits_prebinning)
            else:   
                n_buckets = len(self._splits_prebinning) + 1

            self._n_prebuckets = n_buckets

            # retrieve information from the binning table
            nonevent = ivt.iloc[:, 2].values[:n_buckets]
            event = ivt.iloc[:, 3].values[:n_buckets]           
            default = ivt.iloc[:, 4].values[:n_buckets]

        # save IV after cleaning ivt
        self._iv_prebinning = ivt.IV.values[-1]

        return nonevent, event, default

    def _optimize(self, x, y, sample_weight=None):
        """Build MIPCL model and solve."""
        start = time.perf_counter()

        # compute optimization problem data
        nev, ev, min_bucket_size, max_bucket_size = self._pre_optimize_data(
            x, y, sample_weight)

        # no need to solve optimization problem, no buckets.
        if self._n_prebuckets in [0, 1]:
            if self.verbose:
                print("optimization: prebuckets <= 1, no optimization",
                    "is required.")

            self._is_solution = True
            self._is_solution_optimal = True
            self._splits_optimal = self._splits_prebinning

            # add group others if generated
            if self._group_others:
                self._splits_optimal = list(self._splits_optimal) + [
                    self._splits_others]

            self._infeasible_buckets = 0
        else:
            # initialize optimization problem
            self._mipcl_problem = MIPCLSolver("OptBin")

            if self.monotonicity_sense in ("peak", "valley"):
                if self.verbose:
                    print("pre-binning analysis: trend_change position =", 
                        self._trend_change)
                self._mipcl_problem.model_2(nev, ev, self.monotonicity_sense, 
                    min_bucket_size, max_bucket_size, self.min_buckets, 
                    self.max_buckets, self.min_er_diff,
                    self.user_idx_forced_splits, self.regularization, 
                    self.reduce_bucket_size_diff, self._trend_change,
                    self.max_pvalue)
            else:
                self._mipcl_problem.model(nev, ev, self.monotonicity_sense, 
                    min_bucket_size, max_bucket_size, self.min_buckets, 
                    self.max_buckets, self.min_er_diff,
                    self.user_idx_forced_splits, self.regularization, 
                    self.reduce_bucket_size_diff, self.max_pvalue)

            # run solver and catch C++ std output for post-analysis
            self._run_optimizer()

            # retrieve problem solution and preprocessing information
            solution = self._mipcl_problem.solution()
            self._infeasible_buckets = self._mipcl_problem.infeasible_buckets()

            try:
                if self.dtype in ("numerical", "ordinal"):
                    for idx, v in enumerate(solution[:-1]):
                        if v:
                            self._splits_optimal.append(
                                self._splits_prebinning[idx])
                else:
                    add_bucket = []
                    for idx, v in enumerate(solution):
                        if v:
                            if add_bucket:
                                new_bucket = sum([add_bucket, 
                                    list(self._splits_prebinning[idx])], [])
                                self._splits_optimal.append(np.array(new_bucket))
                                add_bucket = []
                            else:
                                self._splits_optimal.append(
                                    self._splits_prebinning[idx])
                        else:
                            add_bucket += list(self._splits_prebinning[idx])
            except Exception as e:
                # TODO: change method to capture this type of errors.
                # user should be aware of data types, otherwise run block
                # using optimalgrouping.
                print("optimize: check variable type and/or pre-binning algorithm.")
                raise

            # add group others if generated
            if self._group_others:
                self._splits_optimal += [self._splits_others]

            # post-solve information
            self._is_solution = self._mipcl_problem.is_solution
            self._is_solution_optimal = self._mipcl_problem.is_solutionOptimal
            self._is_infeasible = self._mipcl_problem.is_infeasible
            self._is_unbounded = self._mipcl_problem.is_unbounded

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
        # however, as aforementioned this would not work in Jupyter. By default,
        # fileno = 1 ==> standard output.
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

    def _statistics(self, x, y, sample_weight=None):
        """Collect OptBin and MIPCL statistics and perform p-value tests."""
        start = time.perf_counter()

        if self._n_prebuckets >= 2:
            stats = self._parser_solver_output()

            # problem statistics
            self._ncs = stats[0]
            self._nvs = stats[1]
            self._nnz = stats[2]
        else:
            self._ncs = self._n_prebuckets
            self._nvs = self._n_prebuckets
            self._nnz = self._n_prebuckets

        if self._is_solution_optimal and self._n_prebuckets >= 2:

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
            self._time_problem_generation -= self._time_problem_data

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
            self._time_problem_generation -= self._time_problem_data

        # optimal-binning results
        if self.dtype in ("numerical", "ordinal"):
            self._n_optimal_buckets = len(self._splits_optimal) + 1
        else:
            self._n_optimal_buckets = len(self._splits_optimal)

        self._binning_table_optimal, self._splits_specials = table(x, y, 
            self._splits_optimal, sample_weight, self.special_handler_policy, 
            self.special_values, self.special_woe_policy,
            self.missing_woe_policy)
        
        self._iv_optimal = self._binning_table_optimal.IV.values[-1]

        # p-value computation
        if self._is_solution_optimal and len(self._splits_optimal):
            tb = self._binning_table_optimal.iloc[:, 
                [2,3]].values[:self._n_optimal_buckets]

            if self.pvalue_method is "Chi2":
                if len(tb) == 1:
                    self._max_pvalue = 0
                else:
                    self._max_pvalue = max(chi2_contingency(tb[i:i+2], 
                        correction=False)[1] for i in range(len(tb)-1))
            elif self.pvalue_method is "Fisher":
                if len(tb) == 1:
                    self._max_pvalue = 0
                else:
                    self._max_pvalue = max(fisher_exact(tb[i:i+2])[1] 
                        for i in range(len(tb)-1))
        else:
            self._max_pvalue = 0

        # buckets size information
        if self._is_solution_optimal:
            if self._n_prebuckets:
                records = self._binning_table_optimal.iloc[:, 
                    1].values[:self._n_optimal_buckets]
            else:
                records = self._binning_table_optimal.iloc[:, 
                    1].values[:self._n_optimal_buckets+1]

            self._largest_bucket_size = np.max(records)
            self._smallest_bucket_size = np.min(records)
            self._largest_bucket_perc = self._largest_bucket_size / self._n_samples
            self._smallest_bucket_perc = self._smallest_bucket_size / self._n_samples
            self._diff_largest_smallest_bucket_size = self._largest_bucket_size
            self._diff_largest_smallest_bucket_size -= self._smallest_bucket_size
            self._std_bucket_size = np.std(records)

        else:
            self._largest_bucket_size = self._smallest_bucket_size = 1.0
            self._diff_largest_smallest_bucket_size = 0.0
            self._std_bucket_size = 0.0

        # missing / specials and others buckets
        records = self._binning_table_optimal.iloc[:, 1].values[:-1]

        if self._n_prebuckets:
            records_specials = records[self._n_optimal_buckets:-1]
        else:
            records_specials = records[self._n_optimal_buckets+1:-1]

        records_missing = records[-1]
        self._group_special = sum(1 for w in records_specials if w)
        self._group_missing = 1 if records_missing else 0

        self._time_post_analysis = time.perf_counter() - start

    def _parser_solver_output(self):
        """Parser for MIPCL output."""
        lines = self._mipcl_msg.split("\n")

        is_start_pre = False
        is_after_pre = False
        is_optimizing = False
        is_generating_cuts = False
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
                else: continue
                
            if "After preprocessing" in line:
                if not is_after_pre:
                    data = regex_basic.findall(line)
                    ncs_after = int(data[0])
                    nvs_after = int(data[1])
                    nnz_after = int(data[4])
                    is_after_pre = True
                else: continue
                
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
        """
        Report OptBin statistics. NOTE: output should be improved in future
        releases, current approach is hard to maintain.
        """
        prebinning = "yes" if self._is_prebinning else "no"
        minbs = "not set" if self.min_buckets is None else self.min_buckets
        maxbs = "not set" if self.max_buckets is None else self.max_buckets 

        if self.monotonicity_sense is "ascending":
            sense = "asc"
        elif self.monotonicity_sense is "descending":
            sense = "desc"
        elif self.monotonicity_sense is "concave":
            sense = "concave"
        elif self.monotonicity_sense is "convex":
            sense = "convex"
        elif self.monotonicity_sense is "peak":
            sense = "peak"
        elif self.monotonicity_sense is "valley":
            sense = "valley"
        elif self.monotonicity_sense is "undefined":
            sense = "undefined"

        if self.monotonicity_user is "ascending":
            user_sense = "asc"
        elif self.monotonicity_user is "descending":
            user_sense = "desc"
        elif self.monotonicity_user is "concave":
            user_sense = "concave"
        elif self.monotonicity_user is "convex":
            user_sense = "convex"
        elif self.monotonicity_user is "peak":
            user_sense = "peak"
        elif self.monotonicity_user is "valley":
            user_sense = "valley"           
        else:
            user_sense = "auto"

        spec = "yes" if self.special_values else "no"
        prebuckets = "yes" if self.user_splits_provided else "no"
        indexforced = "yes" if self.user_idx_forced_splits else "no"
        regularization = "yes" if self.regularization else "no"
        reduce_bucket_diff = "yes" if self.reduce_bucket_size_diff else "no"
        max_pvalue = self.max_pvalue if self.max_pvalue is not None else "no"

        # pre-binning algorithm options
        if self.prebinning_algorithm is "ctree":
            ct_vartype = self.dtype
            ct_mincrit = round(self._ctree_min_criterion, 8)
            ct_maxcand = self.ctree_max_candidates
            ct_dynamic = self.ctree_dynamic_split_method
        else:
            ct_vartype = "not set"
            ct_mincrit = "not set"
            ct_maxcand = "not set"
            ct_dynamic = "not set"

        if not self._is_prebinning:
            pre_algo = "not set"
        else:
            pre_algo = self.prebinning_algorithm
        
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

        # buckets reduction
        buckets_reduction = self._n_optimal_buckets - self._n_prebuckets

        if self._n_prebuckets >= 2:
            infeas_ratio = self._infeasible_buckets / self._n_prebuckets
        else:
            infeas_ratio = 0

        # IV loss
        iv_loss = (self._iv_optimal - self._iv_prebinning) / self._iv_prebinning

        report = (
        "\033[94m================================================================================\033[0m\n"
        "\033[1m\033[94m                  GRMlab OptBin 0.1: Feature binning optimizer                  \033[0m\n"
        "\033[94m================================================================================\033[0m\n"
        "\n"
        " \033[1mMain options                              Extra options\033[0m\n"
        "   pre-binning              {:>4}             special values               {:>3}\n"
        "   pre-binning max nodes    {:>4}             special handler policy{:>10}\n"
        "   monotonicity sense    {:>7}             special WoE policy    {:>10}\n"
        "   min buckets           {:>7}             missing WoE policy    {:>10}\n"
        "   max buckets           {:>7}\n"
        "   min bucket size         {:>4.3f}           \033[1mUser pre-binning options\033[0m\n"
        "   max bucket size         {:>4.3f}             pre-buckets                  {:>3}\n"
        "   min PD difference       {:>4.3f}             indexes forced               {:>3}\n"
        "   regularization           {:>4}\n"
        "   reduce bucket size diff  {:>4}\n"
        "   max p-value              {:>4}\n"
        "\n"
        " \033[1mPre-binning algorithmic options\033[0m\n"
        "   algorithm             {:>7}\n"
        "   ctree options\n"
        "     variable type  {:>12}\n"
        "     min criterion      {:>8}\n"
        "     max candidates      {:>7}\n"
        "     DSM          {:>14}\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        "\n"
        " \033[1mProblem statistics                        Optimizer statistics\033[0m\n"
        "   original problem                          status                {:>10}\n"
        "     variables          {:>6}               objective           {:>12.5f}\n"
        "     constraints        {:>6}               preprocessor\n"
        "     nonzeros           {:>6}                 infeasible buckets {:>4} ({:>4.0%})\n"
        "   after preprocessing                       cutting planes\n"
        "     variables          {:>6} ({:>7})       cuts generated     {:>4}\n"
        "     constraints        {:>6} ({:>7})       cuts used          {:>4} ({:>4.0%})\n"
        "     nonzeros           {:>6} ({:>7})     branch-and-cut nodes {:>4}\n"
        "\n"
        " \033[1mTiming statistics                         Optimizer options\033[0m\n"
        "   total                {:>6.3f}               root LP algorithm   {:>10}\n"
        "     prebinning         {:>6.3f} ({:>4.0%})        time limit            {:>10}\n" 
        "     model data         {:>6.3f} ({:>4.0%})        MIP gap                 {:>7.6f}\n"
        "     model generation   {:>6.3f} ({:>4.0%})                                        \n"
        "     optimizer          {:>6.3f} ({:>4.0%})                                        \n"
        "       preprocessing      {:>6.3f} ({:>4.0%})                                      \n"
        "     post-analysis      {:>6.3f} ({:>4.0%})                                        \n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        "\n"
        " \033[1mPre-binning statistics                    Optimal-binning statistics\033[0m\n"
        "   buckets              {:>6}               buckets               {:>3} ({:>4})\n"
        "   IV                  {:>7.5f}               IV                {:>7.5f} ({:>4.0%})\n"
        "   trend changes        {:>6}               monotonicity           {:>9}\n"
        "                                             p-value ({:>6})         {:>6.5f}\n"
        "\n"
        "                                             largest bucket     {:>6} ({:>4.0%})\n"
        "                                             smallest bucket    {:>6} ({:>4.0%})\n"
        "                                             std bucket size       {:>10.2f}\n"
        "   \033[94m--------------------------------------------------------------------------\033[0m\n"
        ).format(prebinning, spec, self.rtree_max_leaf_nodes, 
        self.special_handler_policy, user_sense, self.special_woe_policy, 
        minbs, self.missing_woe_policy, maxbs, self.min_bucket_size, 
        self.max_bucket_size, prebuckets, self.min_er_diff, indexforced,
        regularization, reduce_bucket_diff, max_pvalue,
        pre_algo, ct_vartype, ct_mincrit, ct_maxcand, ct_dynamic,
        status, self._nvs, self._mipcl_obj, self._ncs, self._nnz, 
        self._infeasible_buckets, infeas_ratio, self._nvs_preprocessing, 
        self._nvs_removed, cut_g, self._ncs_preprocessing, self._ncs_removed, 
        cut_u, cut_p, self._nnz_preprocessing, self._nnz_removed, 
        self._branch_and_cut_nodes, self._time_total, "Dual-Simplex", 
        self._time_prebinning, self._time_prebinning/self._time_total, 
        self._time_limit, self._time_problem_data, 
        self._time_problem_data/self._time_total, 0, 
        self._time_problem_generation,
        self._time_problem_generation/self._time_total,
        self._time_optimizer, self._time_optimizer/self._time_total,
        self._time_optimizer_preprocessing, rel_pre, self._time_post_analysis, 
        self._time_post_analysis/self._time_total, self._n_prebuckets, 
        self._n_optimal_buckets, buckets_reduction, self._iv_prebinning, 
        self._iv_optimal, iv_loss, self._prebinning_trend_changes, sense, 
        self.pvalue_method, self._max_pvalue, 
        self._largest_bucket_size, self._largest_bucket_perc,
        self._smallest_bucket_size, self._smallest_bucket_perc, 
        self._std_bucket_size)

        print(report)

    def _compute_trend_change(self, default_rate):
        """Compute trend change split for valley/peak non-rigorous approach."""
        if self.monotonicity_sense is "peak":
            self._trend_change = default_rate.argmax()
        elif self.monotonicity_sense is "valley":
            self._trend_change = default_rate.argmin()

    def _monotonicity_parameters(self, nonevent, event, default_rate):
        """
        Compute parameters needed by the decision tree to determine the correct
        monotonicity sense when "auto" mode is active.
        """

        # curve-fitting and quadratic improvement
        x = np.arange(len(default_rate), dtype=float)
        default_rate = np.asarray(default_rate, dtype=float)
        linear = np.polyfit(x, default_rate, deg=1, full=True)
        quadratic = np.polyfit(x, default_rate, deg=2, full=True)

        linear_residual = linear[1]
        quadratic_residual = quadratic[1]

        diff_fitting = abs(linear_residual - quadratic_residual)
        quadratic_improvement = diff_fitting / linear_residual

        linear_coef = linear[0][0]
        quadratic_coef = quadratic[0][0]
        linear_sense = "descending" if linear_coef < 0 else "ascending"
        quadratic_sense = "peak" if quadratic_coef < 0 else "valley"

        # triangle area: abs(0.5 * det[A]))
        x0 = default_rate[0]; y0 = 0
        xn = default_rate[-1]; yn = default_rate.size
        xmin = default_rate.min(); ymin = default_rate.argmin()
        xmax = default_rate.max(); ymax = default_rate.argmax()

        Amin = np.array([[x0, xmin, xn], [y0, ymin, yn], [1, 1, 1]])
        Amax = np.array([[x0, xmax, xn], [y0, ymax, yn], [1, 1, 1]])
        area_min = 0.5 * abs(np.linalg.det(Amin))
        area_max = 0.5 * abs(np.linalg.det(Amax))   

        # dominant triangle
        area = max(area_min, area_max)
        area_total = (xmax-xmin) * yn
        area_ratio = area / area_total

        # percentage at both sides of trend breakpoint
        if area == area_min: # area = are_min => valley
            left = sum(nonevent[:ymin]) + sum(event[:ymin])
            right = sum(nonevent[ymin+1:]) + sum(event[ymin+1:])

            left_elements = nonevent[:ymin] + event[:ymin]
            right_elements = nonevent[ymin+1:] + event[ymin+1:]

            left_mean = np.mean(left_elements) if len(left_elements) else 0
            right_mean = np.mean(right_elements) if len(right_elements) else 0          
        else: # area = are_max => peak
            left = sum(nonevent[:ymax]) + sum(event[:ymax])
            right = sum(nonevent[ymax+1:]) + sum(event[ymax+1:])

            left_elements = nonevent[:ymax] + event[:ymax]
            right_elements = nonevent[ymax+1:] + event[ymax+1:]

            left_mean = np.mean(left_elements) if len(left_elements) else 0
            right_mean = np.mean(right_elements) if len(right_elements) else 0

        # mean_left/mean and mean_right/mean
        mean_total = np.mean(nonevent + event)
        mean_left_mean = left_mean / mean_total
        mean_right_mean = right_mean / mean_total

        # compute convex hull
        len_pds = len(default_rate)
        points = np.zeros((len_pds, 2))
        points[:, 0] = np.arange(len_pds)
        points[:, 1] = default_rate

        total_area = (xmax - xmin) * yn

        if len_pds > 2:
            try:
                hull = ConvexHull(points)
                convexhull_area = hull.volume / total_area
            except:
                convexhull_area = 0
        else:
            convexhull_area = 0

        # trend change peak/valley
        if quadratic_sense is "peak":
            self._trend_change = ymax
        elif quadratic_sense is "valley":
            self._trend_change = ymin

        return [linear_sense, quadratic_sense, quadratic_improvement, area, 
            left, right, mean_left_mean, mean_right_mean, convexhull_area]

    def _monotonicity_decision(self, linear_sense, quadratic_sense, 
        quadratic_improvement, triangle_area, left, right, mean_left_mean, 
        mean_right_mean, convexhull_area):
        """
        Decision tree to decide which monotonicity sense is most sensible to 
        maximize IV.

        monotonicity: 
            + ascending / descending = 1
            + convex / concave or peak / valley = 0

        Note: as the number of cases increases the algorithm shall improve
        its predictive power.
        """
        monotonicity_sense = 0

        if right <= 0.076:
            if convexhull_area <= 0.473:
                monotonicity_sense = 1
            else:  # if convexhull_area > 0.47283774614334106
                monotonicity_sense = 0
        else:  # if right > 0.07573056221008301
            if mean_left_mean <= 0.778:
                if quadratic_improvement <= 0.058:
                    monotonicity_sense = 1
                else:  # if quadratic_improvement > 0.05759747326374054
                    monotonicity_sense = 0
            else:  # if mean_left_mean > 0.7778470516204834
                monotonicity_sense = 0

        # analyze monotonicity and print result
        if monotonicity_sense == 1 or self.monotonicity_force:
            return linear_sense  # asc / desc
        else:
            # peak/valley and self._iv_prebinning >= 0.05
            return quadratic_sense

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

    @staticmethod
    def _trend_changes(x):
        """
        Detect the number of trend changes from the PD curve computed after
        performing pre-binning.
        """
        n = len(x)
        n1 = n-1

        n_asc = n_des = 0
        peaks = valleys = 0

        for i in range(1, n1):
            if x[i] < x[i-1] and x[i] < x[i+1]:
                valleys += 1
            if x[i] > x[i-1] and x[i] > x[i+1]:
                peaks += 1
        changes = peaks + valleys

        return changes

    @staticmethod
    def _plot_solver_progress(time, obj, cuts):
        """
        Plot MIPCL progress in terms of objective function and cut generation.
        """
        fig, ax = plt.subplots(1,1)
        ax.plot(time, obj, '-xb', label="objective")
        ax2 = ax.twinx()
        ax2.plot(time, cuts, '^c', label="generated cuts")

        # label
        ax.set_ylabel("Objective value")
        ax2.set_ylabel("Cuts")
        ax.set_xlabel("time (s) - after preprocessing")

        # legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
        ax2.legend(lines, labels)

        plt.show()

    @property
    def splits_optimal(self):
        """
        OptBin splits points.

        Returns
        -------
        splits : numpy.ndarray
        """
        return np.asarray(self._splits_optimal)

    @property
    def splits_prebinning(self):
        """
        Prebinning splits points.

        Returns
        -------
        splits : numpy.ndarray
        """
        return np.asarray(self._splits_prebinning)
