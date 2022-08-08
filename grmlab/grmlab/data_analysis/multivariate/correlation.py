"""
Mutlivariate correlations
"""

# Author: Fernando Gallego-Marcos <fernando.gallego.marcos@bbva.com>
# BBVA - Copyright 2019.


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.sparse.linalg import eigsh
from sklearn.manifold import smacof

from ...core.base import GRMlabBase
from ...core.exceptions import NotRunException
from ...modelling.model_analysis.analyzer import basic_statistics


def _group_generator(df, min_corr, name="group"):
    # compute nodes and links
    true_links = (np.abs(df) > min_corr).values
    node_names = df.columns.values
    vec_links = []
    for i, vec in enumerate(true_links):
        for j in range(i):
            if i != j and true_links[i][j]:
                vec_links.append((node_names[i], node_names[j]))

    # graph generator
    G = nx.Graph()
    G.add_nodes_from(node_names)
    G.add_edges_from(vec_links)

    # calculates connected components
    groups = list(nx.connected_components(G))

    dict_groups = {"no_group": []}
    idx_group = 0
    for grp in groups:
        grp = list(grp)
        if len(grp) > 1:
            dict_groups[name + "_" + "{:02d}".format(idx_group)] = grp
            idx_group += 1
        else:
            dict_groups["no_group"].append(grp[0])

    return dict_groups, G


def _max_weight_elt(dict_vars, vars_vec):
    idx_max = np.argmax([dict_vars[var]["weight"] for var in vars_vec])
    return vars_vec[idx_max]


class MultivariateCorrelations(GRMlabBase):
    """Study of the correlations among the variables.

    Parameters
    ----------
    name : str
        Name given to the class. This is only an identifier set by the
        user.

    vars_metrics : dict (default=None)
        Dictionary where the keys are the name of the columns. The value
        is other dictionary with the name of the metric(s) and its value(s).

    metric_weights : dict (default=None)
        Dictionary where the keys are the name of the metrics used to calculate
        the weights of each column. The value is the weight given to each
        metric to calculate the total weight of the variables.

    correlation_method : str (default="pearson")
        Method to calculate the correlation. The options are: 'pearson',
        'kendall', and 'spearman'.

    equal_mincorr : float (default=0.99)
        Minimum correlation between two variables to consider them equal.

    groups_mincorr : float (default=0.6)
        Minimum correlation to link two variables in the same group.

    links_mincorr : float (default=0.4)
        Minimum correlation to display a link between two variables in the
        reporting.

    verbose : boolean (default=True)
        Controls verbosity of output.
    """
    def __init__(self, name, vars_metrics=None, metric_weights=None,
                 correlation_method="pearson", equal_mincorr=0.99,
                 groups_mincorr=0.6, links_mincorr=0.4, verbose=True):

        # general parameters
        self.name = name
        self.vars_metrics = vars_metrics
        self.metric_weights = metric_weights
        self.correlation_method = correlation_method
        self.equal_mincorr = equal_mincorr
        self.groups_mincorr = groups_mincorr
        self.links_mincorr = links_mincorr
        self.verbose = verbose

        # data
        self._n_samples = None
        self._n_columns = None

        # run
        self._n_samples = None
        self._column_names = []
        self._column_distinct = []
        self._dict_variables = {}
        self._correlation_metrics = None
        self.mds = None

        self.df_corr = None
        self.df_corr_distinct = None
        self.df_euclidean_distance = None

        self._groups_equal = []
        self.graph_equal = None
        self._groups = {}
        self.graph = None

        # flags
        self._is_run = False

        if ((self.metric_weights is not None) and
                (self.vars_metrics is not None)):
            self._flag_weights = True
        else:
            self._flag_weights = False

    def results(self):
        """
        Return information and flags for all variables in the multivariate
        correlation analysis.
        """
        return pd.DataFrame.from_dict(
            self._dict_variables,
            columns=["group_id", "connections", "weight", "excluded",
                     "excluded_by"],
            orient="index").reset_index().rename(columns={"index": "name"})

    def plot_distances(self):
        """Plot the variables positions in a 2D plot according to the
        classical scaling algorithm results.

        The method used for the classical scaling can be found in
        :cite:`Izenman`.
        """

        if not self._is_run:
            raise NotRunException(self, "run")

        plt.scatter(self.mds[:, 0], self.mds[:, 1], alpha=0.6)
        for i, txt in enumerate(self._column_distinct):
            plt.annotate(txt, (self.mds[i, 0], self.mds[i, 1]))
        plt.show()
        plt.close()

    def run(self, data):
        """Run the correlation analysis.

        Parameters
        ----------
        data : pandas.DataFrame
            Database used to generate the correlation graph.

        Returns
        -------
        self.graph_equal : object networkx.classes.graph.Graph
            Graph where any pair of nodes which have a direct edge connecting
            them are considered the same variable.

        self.graph : object networkx.classes.graph.Graph
            Graph where any group of nodes connected directly or indirectly
            among them are considered from the same group.
        """
        self._n_samples = data.shape[0]
        self._n_columns = data.shape[1]
        self._column_names = data.columns.values

        if self._flag_weights:
            for metric in self.metric_weights:
                for var in self._column_names:
                    if var not in self.vars_metrics:
                        raise KeyError("variable '{}' not included in "
                                       "vars_metrics".format(var))
                    if metric not in self.vars_metrics[var]:
                        raise KeyError(
                            "Metric '{}' not given for variable '{}'".format(
                                metric, var))

        # check equal_mincorr, groups_mincorr, links_mincorr
        if not (self.equal_mincorr >= 0 and self.equal_mincorr <= 1):
            raise ValueError("equal_mincorr must be in range (0,1)")
        if not (self.groups_mincorr >= 0 and self.groups_mincorr <= 1):
            raise ValueError("groups_mincorr must be in range (0,1)")
        if not (self.links_mincorr >= 0 and self.links_mincorr <= 1):
            raise ValueError("links_mincorr must be in range (0,1)")

        self._run(data)

        return self.graph_equal, self.graph

    def _run(self, data):

        # init variable dictionary
        for var in self._column_names:
            if self._flag_weights:
                metric_val = sum(
                    self.vars_metrics[var][key]*self.metric_weights[key]
                    for key in self.metric_weights)
            else:
                metric_val = 1
            self._dict_variables[var] = {
                "group_id": None, "excluded": False, "excluded_by": None,
                "corr": None, "distance": None, "mds": None,
                "weight": metric_val, "connections": 0}

        # correlations
        if self.correlation_method == "pearson":
            corr = np.corrcoef(data.values.T)
            self.df_corr = pd.DataFrame(
                corr, columns=data.columns, index=data.columns)
        else:
            self.df_corr = data.corr(method=self.correlation_method)

        if self._flag_weights:
            # detection of groups with equal variables
            self._groups_equal, self.graph_equal = _group_generator(
                self.df_corr, self.equal_mincorr, name="equal")

            # get the disconnected variables with highest weight from each
            # group
            self._column_distinct = []
            for key in self._groups_equal:
                if key == "no_group":
                    continue
                keep = []  # variables that are not removed
                remove = []  # variables to exclude
                processed = []  # keep track of the processed variables
                not_processed = np.array(self._groups_equal[key])
                while len(not_processed) > 1:
                    # get and keep the variable with highest weight.
                    keep.append(_max_weight_elt(self._dict_variables,
                                                not_processed))
                    # remove the variables directly connected with the variable
                    # with highest weight.
                    _remove = [tpl[1] for tpl
                               in self.graph_equal.edges(keep[-1])
                               if tpl[1] not in processed]
                    remove += _remove
                    processed = remove + keep
                    not_processed = not_processed[
                        [(elt not in processed) for elt in not_processed]]
                    # add removal info to dict_variables
                    for var in _remove:
                        self._dict_variables[var]["excluded"] = True
                        self._dict_variables[var]["excluded_by"] = keep[-1]
                self._column_distinct += keep + list(not_processed)

            self._column_distinct += self._groups_equal["no_group"]
        else:
            self._column_distinct = self._column_names

        # filter the variables set as remove.
        self.df_corr_distinct = self.df_corr[
            self._column_distinct].loc[self._column_distinct]

        # groups detection
        self._groups, self.graph = _group_generator(self.df_corr_distinct,
                                                    self.groups_mincorr)
        for key in self._groups:
            for var in self._groups[key]:
                self._dict_variables[var]["group_id"] = key
        removed_vars = (set(self._column_names) - set(self._column_distinct))
        for var in removed_vars:
            removed_by_var = self._dict_variables[var]["excluded_by"]
            self._dict_variables[var][
                "group_id"] = self._dict_variables[removed_by_var]["group_id"]

        # Euclidean distance from correlations divided by the number of
        # samples
        self.df_euclidean_distance = 2 * (1 - np.abs(self.df_corr_distinct))

        # multidimensional scaling algorithm
        # self.mds = self._classical_scaling_algorithm()
        self.mds = smacof(self.df_euclidean_distance, metric=True)[0]

        # save to dict
        for i, var in enumerate(self._column_distinct):
            self._dict_variables[var]["corr"] = list(
                self.df_corr_distinct[var].values)
            self._dict_variables[var]["distance"] = list(
                self.df_euclidean_distance[var].values)
            self._dict_variables[var]["mds"] = list(list(np.real(self.mds[i])))
            self._dict_variables[var]["connections"] = sum(
                np.array(self._dict_variables[var]["corr"]) >
                self.groups_mincorr) - 1

        # correlation statistics
        corr = (self.df_corr_distinct -
                np.diag(np.diag(self.df_corr_distinct))).values
        if corr.shape[0] > 2:
            correlations_vector = np.abs(corr[np.tril(corr) != 0.])
            self._correlation_metrics = basic_statistics(correlations_vector)

        self._is_run = True

    def _classical_scaling_algorithm(self, n=2):
        # square distances
        A = - np.power(np.array(self.df_euclidean_distance), 2) / 2
        nA = len(A)
        H = np.diag(np.ones(nA)) - np.full((nA, nA), 1/nA)
        # B matrix is b_ij = a_ij + mean(a_ij) - man(col_j) - mean(row_i)
        B = np.matmul(np.matmul(H, A), H)
        Lambda, V = eigsh(B, n, which='LM')
        Y = np.multiply(V, np.sqrt(Lambda))
        return Y
