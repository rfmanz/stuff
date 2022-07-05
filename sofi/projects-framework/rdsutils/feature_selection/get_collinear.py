import numpy as np
import pandas as pd
import networkx as nx

def get_collinear_(df, features, corr_lowerbound=-1, corr_upperbound=1,
                  num_features=None):
    """
    @returns collinear_mtx: collinear matrix containing high values
    @returns to_drop: high correlation pairs
    """
    if num_features is not None:
        data = df[num_features]
    else:
        data = df[features]._get_numeric_data()

    corr = data.corr()
    corr_ = corr.copy(deep=True)
    n = len(corr)

    # highly correlated
    corr_ = corr_[(corr_ <= corr_lowerbound) |
                  (corr_ >= corr_upperbound)]
    corr_.iloc[range(n), range(n)] = np.nan  # mask diagonals
    corr_.dropna(axis=0, how='all', inplace=True)
    corr_.dropna(axis=1, how='all', inplace=True)
    rows, cols = corr_.index, corr_.columns

    to_drop = []
    for i in range(len(corr.index)):  # these two should be the same
        for j in range(i+1, len(corr.columns)):
            row = corr.index[i]
            col = corr.columns[j]
            if (not np.isnan(corr.loc[row, col])
                and (row != col)
                and (corr.loc[row, col] <= corr_lowerbound
                     or corr.loc[row, col] >= corr_upperbound)):
                to_drop.append((row, col, corr.loc[row, col]))
                
    collinear_mtx = corr.loc[rows, cols]
    return collinear_mtx, to_drop 


################################################
#           Remove collinear features
################################################


def get_collinear_graph(collinear_pairs):
    """
    fsel.record_collinear_pairs
    """
    import networkx as nx
    
    
    G = nx.Graph()
    G.add_weighted_edges_from(collinear_pairs)
    
    return G


def plot_collinear_graph(G, figsize=(6,5), layout_seed=12345):
    import networkx as nx
    
    fig = plt.figure(figsize=figsize)
    try:
        pos = nx.planar_layout(G)
    except:
        pos = nx.random_layout(G, seed=layout_seed)  # nx.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos, alpha=0.4)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Collinear Features Graph')
    plt.tight_layout()
    plt.show()
    
    return fig


def get_degree_df(G, ascending_degree=False):
    degree_df = pd.DataFrame(G.degree)
    degree_df.columns = ['feature', 'degree']
    degree_df.sort_values('degree',
                          ascending=ascending_degree, 
                          inplace=True)
    return degree_df


def get_node_tot_corr(G, node):
    ngbrs = nx.all_neighbors(G, node)
    size = 0
    for n in ngbrs:
        # take abs value of the negative corr
        size += abs(G.get_edge_data(node, n)['weight'])
    return size


def remove_node_by_sum_corr(G, removal_order, degree_df):
    """
    @returns graph, feature_to_drop
    """
    # get candidate cluster to drop
    # choose the cluster center with highest sum of correlation

    # drop by sum correlation...
#     cands = degree_df[degree_df.degree == degree_df.degree.max()]
#     cands.loc[:, 'weights_sum'] = 0
#     for i, row in cands.iterrows():
#         cur_f = row['feature']
#         cands.loc[i, 'weights_sum'] = get_node_tot_corr(G, cur_f)

#     # take the neighbor of the top candidate
#     idx = cands['weights_sum'].argmax()
#     cand_feature = cands.loc[idx, 'feature']
    
    raise NotImplementedError 
        
    
def remove_node_by_order(G, removal_order, degree_df):
    features = degree_df[degree_df.degree>0].feature
    to_drop_idx = min([removal_order.index(f) for f in features])
    to_drop = removal_order[to_drop_idx]
    return to_drop


def remove_nodes_until_empty_graph(G, removal_order, drop_strategy='low_importance'):
    """
    remove graph nodes until there is no edges - 0 degree nodes
    
    default removal order: 
        - remove the node with highest sum of weights with neightbors
        - highest correlation with the squad
        - use `removal_order` to break ties.  # degree 1 edges
        
    @params G: networkx graph object
    @params removal_order: list of nodes 
                - each node index reflecting its importance
                - e.g. ['a', 'b', 'c'],  imp('c') = 2 > imp('a') = 0
    @params drop_strategy: str - method to drop features
                - 'sum_corr': drop feature with higher sum of corr with neighbors
                - 'low_importance': drop feature with lower feature importance
                
            - from my experience, highly correlated feature may have similar 
              predictive power, but the numbers may not be stable because of
              collinearity
    """
    graph = G.copy()
    degree_df = get_degree_df(graph)
    features_to_drop = []
    
    while (degree_df.degree > 0).any():
        
        if drop_strategy=='sum_corr':
            to_drop = remove_node_by_sum_corr(graph, 
                                             removal_order,
                                             degree_df)
        
        elif drop_strategy=='low_importance':
            to_drop = remove_node_by_order(graph, 
                                          removal_order,
                                          degree_df)
        
        else:
            raise ValueError('drop_strategy must choose ["sum_corr", "low_importance"]')
        
        graph.remove_node(to_drop)
        features_to_drop.append(to_drop)
        degree_df = get_degree_df(graph)
        
    return graph, features_to_drop
        

def get_collinear_features_to_drop_(collinear_pairs, removal_order):
    """
    provide the fsel.record_collinear_pairs, return the features to drop
    
    @params collinear_pairs: fsel.record_collinear_pairs 
                - [(node1, node2, edge weight)]
    @params removal_order: list of nodes 
                - each node index reflecting its importance
                - e.g. ['a', 'b', 'c'],  imp('c') = 2 > imp('a') = 0
    """
    corr_pairs = map(lambda x: (x[0], x[1], round(x[2], 3)), collinear_pairs)
    G = get_collinear_graph(corr_pairs)
    G, to_drop = remove_nodes_until_empty_graph(G, removal_order)
    return to_drop