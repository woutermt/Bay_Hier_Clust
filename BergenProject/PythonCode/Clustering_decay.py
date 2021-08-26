"""
Wouter Tijs
21/10/2019
University Bergen

REFERENCES
(1) Lemon-tree Bonnet E, Calzone L, Michoel T. (2015)
    Integrative multi-omics module network inference with Lemon-Tree. PLoS Comput Biol 11(2): e1003983.
(2) Hagberg, A.A. et al. (2008) Exploring Network Structure, Dynamics, and Function using NetworkX.
    In, Varoquaux,G. et al. (eds), Proceedings of the 7th Python in Science Conference. Pasadena, CA USA, pp. 11–15.
(3) Hunter, J.D. (2007) Matplotlib: A 2D graphics environment. Comput. Sci. Eng., 9, 90–95.


"""

import numpy as np
import pandas as pd
from datetime import date

from itertools import combinations
import random
import networkx as nx

# Pre-process
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata

# Normal-gamma distribution
from scipy.special import gammaln

# PLOTS
from matplotlib import pyplot as plt
import seaborn as sb


######## TEMP ARGS ########
arg_input_file = 'processed_yeast_data_AL_1\linear_regression_expr_var4\expression_statsmodels_linreg_residuals_01.txt'
# arg_input_gene_filtering = '../RawData/processed_yeast_data_AL_1\linear_regression_expr_var4\p2p3_th10pct_gene_interactions.csv'

# # # Manual files for overlap validation
arg_input_gene_filtering = '../Results/input_file_prob_genes.csv'
# manual_gene1 = "YLR264W"
# manual_gene2 = "YLR273C"					# YLR326W (183), YLR273C (), YLR257W, YLR252W

# arg_input_gene_filtering = '../Results/saved_selected_tar_regu_0.999_YLR271W_YLR271W.csv'
# manual_gene1 = "YLR271W"
# manual_gene2 = "YLR254C"

arg_input_path = '../RawData/'
arg_stats_path = '../Stats/'
arg_output_path = '../Results/'
arg_figure_path = '../Graphs/'
arg_outfile_type = 'csv'  					# AdjacencyList: 'adjlist', edgelist: 'edglist', JSON: 'json', GEFX: 'gefx'
arg_graph_type = 'hier'
arg_date_today = date.today()

arg_dendro = False
arg_heatmap = True
arg_allow_correction_score = True
arg_sort = "topological"                        # topological or bayes

arg_filter_method ="manual"			        	# "prob", "rand", "std", "manual"
arg_filt_thres = 0.9998				            # std: 3.6 -- prob: 0.995 is 3709 unique / 0.999 is 1995 unique / 0.9995 is 296 unique / 0.9998 is 39 unique
arg_gamma_prior = [0.1, 0, 0.1, 0.1]  		    # Initialize Normal-Gamma prior lambda, mu, alpha, beta

arg_thes_parent_link = 15
arg_decay = 5
arg_parent_thres = 4

arg_graph_labels = True
arg_g_clean_level = 0
arg_plt_space_x = .5
arg_plt_space_mult_y = 2
arg_plt_label_rota = "45"
arg_plt_label_y = -0.5

arg_output_file = arg_output_path + arg_graph_type + '_' + arg_filter_method + '_threshold_' + str(arg_filt_thres) + '_decay_' + str(arg_decay) + '_' + str(arg_date_today) + '.' + arg_outfile_type
######## TEMP ARGS ########


# CONSTANTS
HALF_LOG_TWO_PI = -0.5 * np.log(2 * np.pi)


def termination_check(termination_condition, graph, gene_merge_score):
    """
    To help terminate the while loop associated with graph construction
    Args:
        termination_condition: Boolean which feeds back into a loop-break
        graph: Networkx graph object
        gene_merge_score: Float score when favourable merge available, otherwise str statement for termination

    Returns: Boolean linked to break statement
    """

    # Terminate if: none of the combinations explored in finding a favourable max_scoring gene is sufficient
    if gene_merge_score == "no favourable merge":
        termination_condition = True

    # Terminate if: every node except the last parent made is a child - fully connected termination condition
    unused_nodes = 0
    for node in graph.nodes:
        if graph.nodes[node]["parent"] is None:
            unused_nodes += 1

    if unused_nodes == 1:
        gene_merge_score = "root found"
        termination_condition = True

    if type(gene_merge_score) == str:
        print("The Graph was terminated because: ", gene_merge_score)

    return termination_condition


def prune_graph(graph, level=None):
    """
    Args:
        graph: Networkx graph object
        level: Graph for level of cleanup node removal, optional with default None

    Returns: Pruned graph
    """

    removed_count = 0

    # Removing all nodes of a certain level to remove irrelevant clusters
    if level is not None:
        for node in list(graph):
            if graph.nodes[node]['level'] <= level:
                if graph.nodes[node]["parent"] is None:
                    for child in graph.nodes[node]["children"]:
                        graph.remove_node(child)
                    graph.remove_node(node)
                    removed_count += 1

    else:
        # Remove all unused nodes, without parent or children from the image
        for node in list(graph):
            if (graph.nodes[node]["children"] == []) and (graph.nodes[node]["parent"] is None):
                graph.remove_node(node)
                removed_count += 1

    print(removed_count, " nodes without a parent removed from graph level threshold ", arg_g_clean_level)

    return graph


def sort_graph(graph, g_data, gene_names, sort):
    """
    Args:
        graph: Networkx graph object
        g_data: Matrix with gene expression values
        gene_names: List with gene ID identifiers
        sort: Argument for graph sorting method: Topological or based on Bayes scores

    Returns: Expression matrix and gene names for re-ordering heatmap configuration
    """

    sorted_g_data = g_data
    sorted_g_names = gene_names

    # Ordering graph for heatmap
    if sort == "topological":
        # Built in topological sorting method
        sorted_graph = nx.topological_sort(graph)

        # Referencing the gene order and applying this to the data
        index_list = []
        for gene in list(sorted_graph):
            if gene[0] != "P":
                index_list.append(np.where(gene_names == gene)[0][0])

        sorted_g_data = g_data[:, index_list]
        sorted_g_names = gene_names[index_list]

    if sort == "bayes":
        sorted_g_names = []
        for a in sorted(graph.nodes(data="pos"), key=lambda x: x[1][0]):
            if a[0][0] != "P":
                sorted_g_names.append(a[0])

        index_list = []
        for gene in list(sorted_g_names):
            index_list.append(np.where(gene_names == gene)[0][0])
        sorted_g_data = g_data[:, index_list]

    return sorted_g_data, sorted_g_names


def draw_graph(graph, figure_path, date_today, filter_method, plt_label_rota, plt_label_y, type='planar', label=False):
    """
    Args:
        graph: Networkx graph object
        figure_path: Path to save location
        date_today: Date today
        filter_method: Filtering method applied for labeling purposes
        plt_label_rota: Label rotation
        plt_label_y: Label y-position relative to X-axis genes
        type: Graph-layout type
        label: Boolean controlling figure labels

    Returns: Preliminary representation of DAG structure in plot
    """
    # Default layout method, overwritten if another is chosen
    pos = nx.circular_layout(graph)

    if type == 'hier':
        pos = nx.get_node_attributes(graph, "pos")

    if type == 'planar':
        try:
            pos = nx.planar_layout(graph)

        except nx.NetworkXException:
            pass

    if type == 'circular':
        pos = nx.circular_layout(graph)

    if type == 'kamada':
        pos = nx.kamada_kawai_layout(graph)

    if type == 'spring':
        pos = nx.spring_layout(graph)

    if type == 'shell':
        pos = nx.shell_layout(graph)

    if type == 'random':
        pos = nx.random_layout(graph)

    if type == 'fruchter':
        pos = nx.fruchterman_reingold_layout(graph)

    if type == 'spiral':
        pos = nx.spiral_layout(graph)

    nx.draw_networkx_nodes(graph, pos, nodelist=[i for i in list(graph) if i[0] == 'P'], node_color='r', label="Parent nodes", node_size=130)
    nx.draw_networkx_nodes(graph, pos, nodelist=[i for i in list(graph) if i[0] != 'P'], node_color='b', label="Original genes/clusters", node_size=130)
    nx.draw_networkx_edges(graph, pos)

    if label :


        pos_labels = {}
        # Re-label at different orientation and create offset
        for gene_name, coord in pos.items():
            gene_score = str(int(graph.nodes[gene_name]["sum_bayes_score"]))
            pos[gene_name] = [coord[0], coord[1] - 0.2]

            # Creating dictionary for re
            label = gene_name + ' (' + gene_score + ')'
            pos_labels[gene_name] = label

        text = nx.draw_networkx_labels(graph, pos, labels=pos_labels, font_size=9)
        for g, t in text.items():
            if g[0] != "P":
                t.set_rotation(plt_label_rota)
                t.set_y(plt_label_y)

    plt.legend(numpoints=1)
    plt.title("Bayesian hierarchical DAG" + ' ' + filter_method)
    plt.axis('off')
    plt.savefig(figure_path + str(date_today) + '_' + str(len(list(graph))) + '_nx_plot.png')

    plt.show()


def write_result(graph, write_method, outfile):
    """
    Args:
        graph: Networkx graph object
        write_method: File output format
        outfile: Path and name for output file

    Returns: Graph written to file
    """

    if write_method == 'gexf':
        nx.write_gexf(graph, outfile)

    if write_method == 'gml':
        gml_format_graph = nx.generate_gml(graph, stringizer=None)
        nx.write_gml(gml_format_graph, outfile)

    if write_method == 'adjlist':
        nx.write_adjlist(graph, outfile)

    if write_method == 'edglist':
        nx.write_edgelist(graph, outfile)

    if write_method == 'xml':
        nx.write_graphml(graph, outfile)

    if write_method == 'csv':

        edgelist = nx.to_pandas_edgelist(graph)
        ndf = pd.DataFrame(edgelist)

        ndf.to_csv(outfile)

    print("File saved in format:", write_method)


def bayes_score(stats, gam_pri):
    """
    Args:
        stats: Sufficient statistics from gene / experiment
        gam_pri: Gamma prior [0.1, 0, 0.1, 0.1] lambda, mu, alpha, beta

    Returns: Bayesian score for provided sufficient statistics from gene / experiment

    Sk(El) =
        − 1/2 * R(0) * log(2π)
        + 1/2 * log(λ0 / λ0 + R(0))
        - log Γ(α0)
        + log Γ(α0 + 1/2 * R(0))
        + α0 * log β0
        − (α0 + 1/2 * R(0)) * log β1
    """

    lamb1 = gam_pri[0] + stats[0]
    alph1 = gam_pri[2] + .5 * stats[0]
    bet1 = gam_pri[3] + .5 * (stats[2] - np.power(stats[1], 2) / stats[0]) + gam_pri[0] * np.power(
        (stats[1] - gam_pri[1] * stats[0]), 2) / (2 * lamb1 * stats[0])

    score = stats[0] * HALF_LOG_TWO_PI + \
            0.5 * np.log(gam_pri[0]) - \
            0.5 * np.log(lamb1) - \
            gammaln(gam_pri[2]) + \
            gammaln(alph1) + \
            gam_pri[2] * np.log(gam_pri[3]) - \
            alph1 * np.log(bet1)

    return score


def merge_stats(node1_stats, node2_stats):
    """
    Args:
        node1_stats: Sufficient statistics from first node to be merged
        node2_stats: Sufficient statistics from second node to be merged

    Returns: Merged sufficient statistics
    """

    # Calculating sufficient statistics
    experiment_count = node1_stats[0] + node2_stats[0]
    expr_sum_prop = node1_stats[1] + node2_stats[1]
    expr_sumsq_prop = node1_stats[2] + node2_stats[2]

    stats_merge = experiment_count, expr_sum_prop, expr_sumsq_prop

    return stats_merge


def update_graph(graph, n1, n2, b_score, experiment_data, cycle, x_space=0.5, y_space_mult=2):
    """
    Args:
        graph: Networkx graph object
        n1: First node
        n2: Second node
        b_score: Sum of bayesian scores
        experiment_data: Updated node data relating to the new merged parent node
        cycle: Node-creation cycle count for naming of new parent
        x_space: Numeric value for X-axis positioning
        y_space_mult: Numeric multiplier to the offset in Y-axis coordinate

    Returns: Graph with new parent with all children assigned a coordinate for eventual plotting, name of new parent
    """

    par_name = ('P' + str(cycle))

    # Updating child: Plot coordinates
    if n1[0] != "P":
        graph.nodes[n1]["pos"] = [cycle, graph.nodes[n1]["level"]]
    if n2[0] != "P":
        graph.nodes[n2]["pos"] = [cycle + x_space, graph.nodes[n1]["level"]]

    # Updating parent: Child list & level & graph coords
    all_children_list = list(set(graph.nodes[n1]['children']).union(set(graph.nodes[n2]['children'])).union({n1, n2}))
    level = max(graph.nodes[n1]['level'], graph.nodes[n2]['level']) + 1
    pos = [(graph.nodes[n1]["pos"][0] + graph.nodes[n2]["pos"][0]) / y_space_mult, level]

    # Add new parent node
    graph.add_node(par_name, experiment_data=experiment_data, sum_bayes_score=b_score, children=all_children_list, decay=None, parent=None, level=level, pos=pos)

    return graph, par_name


def update_graph_params(graph, n_list, n1, n2, par_name, decay_value, parent_threshold):
    """
    Args:
        graph: Networkx graph object
        n_list: Active node list with available merger nodes
        n1: First node
        n2: Second node
        par_name: Name of new parent
        decay_value: Variable for decay
        parent_threshold: Variable for number of allowed parents for a node

    Returns: Graph object updated with new nodes and parameters, active node list with available merger nodes
    """

    # Add/remove for to be merged
    for n in [n1, n2]:
        # Add edges between children and new parent
        graph.add_edge(par_name, n)

        # Add decay to merged nodes
        graph.nodes[n]['decay'] = decay_value

        # Add parent attrib
        if graph.nodes[n]['parent'] is None:
            graph.nodes[n]['parent'] = [par_name]
        else:
            graph.nodes[n]['parent'].append(par_name)

        # Remove the nodes with parent threshold or first level node with a single parent
        if graph.nodes[n]['parent'] is not None:
            if len(graph.nodes[n]['parent']) == parent_threshold or graph.nodes[n]['level'] == 1:
                n_list.remove(n)

    # Remove nodes for decay condition
    for node in n_list:
        # Update the decay for all nodes (where applicable)
        if graph.nodes[node]['decay'] is not None:
            graph.nodes[node]['decay'] -= 1
            # Remove all nodes that have decayed from node list
            if graph.nodes[node]['decay'] <= 0:
                n_list.remove(node)

    return graph, n_list


def calc_score(graph, exp_list, n1, n2, gam_pri, allow_corr=True):
    """
    Args:
        graph: Networkx graph object
        exp_list: List with experiment IDs
        n1: First node
        n2: Second node
        gam_pri: Gamma prior [0.1, 0, 0.1, 0.1] lambda, mu, alpha, beta
        allow_corr: Boolean controlling the correction score

    Returns: Sum of Bayesian and partition scores, and list of all experiment values for new parent node
    """

    sum_merge_bayes = 0
    sum_merge_part = 0
    experiment_data = []

    # Calc variables per experiment
    for exp in range(0, len(exp_list)):

        # Basic parameters
        log_par_n1 = graph.nodes[n1]['experiment_data'][exp][4]
        log_par_n2 = graph.nodes[n2]['experiment_data'][exp][4]
        stats_n1 = graph.nodes[n1]['experiment_data'][exp][2]
        stats_n2 = graph.nodes[n2]['experiment_data'][exp][2]

        # Per experiment merged: Sufficient statistics, Bayes score, Partition score and other attributes
        gene_exp_stats = merge_stats(stats_n1, stats_n2)
        b_score = bayes_score(gene_exp_stats, gam_pri)

        exper_merge_score = b_score - log_par_n1 - log_par_n2
        exper_merge_log_part = b_score + np.log(1 + np.exp(- exper_merge_score))

        # When there is a previous parent to either node, the parents need their score corrected for the overlap
        if allow_corr:
            for node in [n1, n2]:
                if graph.nodes[node]['parent'] is not None:

                    # Variance, precision and means for overlapping nodes
                    variance_n1 = np.power((stats_n1[2] - stats_n1[1]), 2) / stats_n1[0]
                    variance_n2 = np.power((stats_n2[2] - stats_n2[1]), 2) / stats_n2[0]
                    precis_n1 = 1 / variance_n1
                    precis_n2 = 1 / variance_n2
                    mean_n1 = stats_n1[1] / stats_n1[0]
                    mean_n2 = stats_n2[1] / stats_n2[0]

                    # Correction score to be applied to current and previous parent for containing multiple nodes
                    corr_score = precis_n1 * precis_n2 * np.power((mean_n1 - mean_n2), 2) / (precis_n1 + precis_n2)

                    # Correct current score, once for each potentially "shared node"
                    exper_merge_score -= corr_score
                    exper_merge_log_part -= corr_score

                    # Correct previous parent
                    prev_parent = graph.nodes[node]['parent']
                    graph.nodes[prev_parent[0]]['sum_bayes_score'] -= corr_score
                    # print(prev_parent[0], graph.nodes[prev_parent[0]]['sum_bayes_score'], corr_score, graph.nodes[prev_parent[0]]['sum_bayes_score']-corr_score)
                    pre_par_exper_list = list(graph.nodes[prev_parent[0]]['experiment_data'][exp])
                    pre_par_exper_list[3] -= corr_score
                    graph.nodes[prev_parent[0]]['experiment_data'][exp] = pre_par_exper_list

        # Summing the bayes score over all experiments, with combined values from the potentially merged genes
        sum_merge_bayes += exper_merge_score
        sum_merge_part += exper_merge_log_part

        # Combining all stored values for the genes/experiments - g_name // e_name //
        experiment_data.append(([n1, n2], exp_list[exp], gene_exp_stats,
                                exper_merge_score, exper_merge_log_part,
                                [graph.nodes[n1]['experiment_data'][exp][3],
                                 graph.nodes[n2]['experiment_data'][exp][3]]))

    # Conditional: merge if merge score is worse than non_merged scores then invalidate it
    if sum_merge_bayes < (graph.nodes[n1]['sum_bayes_score'] + graph.nodes[n2]['sum_bayes_score']):
        sum_merge_bayes = -np.inf

    return sum_merge_bayes, sum_merge_part, experiment_data


def max_score_pair(graph, n_list, exp_list, gam_pri, allow_correction_score=True):
    """
    Args:
        graph: Networkx graph object
        n_list: Active node list with available merger nodes
        exp_list: List with experiment IDs
        gam_pri: Gamma prior [0.1, 0, 0.1, 0.1] lambda, mu, alpha, beta
        allow_correction_score: Boolean controlling the correction score

    Returns: Optimal two nodes to merge,
             their sum-Bayesian and Partition scores and their experiment lists with scores
    """

    bayes_max = -np.inf
    part_max = None
    n1_max = None
    n2_max = None
    c_graph = graph.copy()
    max_pair_exp_vals = None

    # Node combinations
    for ni, nj in combinations(range(0, len(n_list)), 2):
        n1 = n_list[ni]
        n2 = n_list[nj]

        # Conditional: If no new genes are added there is no point in creating another parent
        if (graph.nodes[n1]["children"] != []) and (graph.nodes[n2]["children"] != []):
            children_n1 = [child for child in graph.nodes[n1]["children"] if child[0] != "P"]
            children_n2 = [child for child in graph.nodes[n2]["children"] if child[0] != "P"]
            if all(child in children_n1 for child in children_n2) or all(child in children_n2 for child in children_n1):
                continue

        # Conditional: merge previously merged nodes again
        if c_graph.nodes[n1]['parent'] is not None and c_graph.nodes[n2]['parent'] is not None:
            parents_n1 = [parent for parent in graph.nodes[n1]["parent"]]
            parents_n2 = [parent for parent in graph.nodes[n2]["parent"]]
            if any(parent in parents_n1 for parent in parents_n2) or any(parent in parents_n2 for parent in parents_n1):
                continue

        # Get score for each node combination from a copy of the graph
        sum_merge_bayes, sum_merge_part, exp_values = calc_score(c_graph, exp_list, n1, n2, gam_pri, allow_correction_score)

        # Find highest scoring node-pair
        if sum_merge_bayes > bayes_max:
            n1_max = n1
            n2_max = n2
            bayes_max = sum_merge_bayes
            part_max = sum_merge_part
            max_pair_exp_vals = exp_values

    if (n1_max is None) or (n1_max is None):
        bayes_max = "no favourable merge"

    res = [n1_max, n2_max, bayes_max, part_max, max_pair_exp_vals]

    return res


def init_nodes(graph, g_data, g_list, e_list, gam_pri):
    """
    Args:
        graph: Networkx graph object
        g_data: Matrix with gene expression values
        g_list: List with gene IDs
        e_list: List with experiment IDs
        gam_pri: Gamma prior [0.1, 0, 0.1, 0.1] lambda, mu, alpha, beta

    Returns: Networkx graph object
    """

    # For all genes in the set
    for ni in range(0, len(g_list)):

        # Creating value sets for all experiments:
        experiment_data = []
        sum_bayes_score = 0

        for nj in range(0, len(e_list)):

            # Calculating sufficient statistics for each experiment
            gene_experiment_stats = [1, g_data[nj, ni], pow(g_data[nj, ni], 2)]

            # Calculating bayes score for each experiment with sufficient statistics
            exp_bayes_score = bayes_score(gene_experiment_stats, gam_pri)
            sum_bayes_score += exp_bayes_score

            # Adding all values to a list to be set to node attribute:
            # gene, experiment, [R0, R1, R2], experiment bayes score, experiment expression value
            experiment_data.append((g_list[ni], e_list[nj], gene_experiment_stats, exp_bayes_score, exp_bayes_score, g_data[nj, ni]))

        # Setting data to networkx graph structure
        graph.add_node(g_list[ni], experiment_data=experiment_data, sum_bayes_score=sum_bayes_score, children=[], decay=None, parent=None, level=1, pos=None)

    return graph


def dendro(g_data, path, date_today, dendro=False):

    # Dendrogram
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    dendrogram(linkage(g_data, method='ward'))

    plt.savefig(path + str(date_today) + '_dendro_pre_cluster.png')
    if dendro:
        plt.show()

    plt.close()


def heat(g_data, g_names, path, date_today, heat=False):

    # Heatmap
    plt.figure(figsize=(10, 7))
    plt.title("Heatmap")
    corr = np.corrcoef(g_data, rowvar=False)

    # Create a mask array where shapes/types are the same
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sb.heatmap(corr, mask=mask, center=0, square=True, linewidths=.5, annot=False, xticklabels=g_names,
               yticklabels=g_names)

    plt.savefig(path + str(date_today) + '_heat_pre_cluster.png')
    if heat:
        plt.show()
    plt.close()


def process_filter(expr_data, gene_list, filt_thres, file_genes, filter_method):
    """
    Args:
        expr_data:
        gene_list:
        filt_thres:
        file_genes:
        filter_method:

    Returns:
    """

    # Add to list if meet the threshold
    prob_gene_set = []

    # Filtering genes on standard deviation based on a threshold
    if filter_method == "std":
        filter_condition = expr_data.std(axis=0) > filt_thres

    # Randomly select a number of genes
    if filter_method == "rand":
        rand_list = []
        for i in range(50):
            rand_list.append(random.randint(0, len(expr_data)))
        filter_condition = rand_list

    # Based on prior research, with gene-set constructed from (Explore_select_prob_data.py)
    if filter_method == "prob" or filter_method == "manual":
        if filter_method == "manual":
            filt_thres = -np.inf
            print("thres", filt_thres)

        # probability matrix to numpy
        gene_selection = pd.read_csv(file_genes)
        gene_list_np = gene_selection.to_numpy()

        for group in gene_list_np:
            if group[2] > filt_thres:
                prob_gene_set += group[0], group[1]

        # Set for unique genes only
        prob_gene_set = set(prob_gene_set)

        # Boolean to select the genes in the set
        for i in range(len(gene_list)):
            filter_condition = [gene in prob_gene_set for gene in gene_list]

    proc_expr_data = expr_data[:, filter_condition]
    proc_gene_list = gene_list[filter_condition]

    # Scale variance in gene expression data
    proc_expr_data = scale(proc_expr_data, axis=0, with_std=True, with_mean=True)

    return proc_expr_data, proc_gene_list


def read_data(input_path, input_file):
    """
    Args:
        input_path: File dir path
        input_file: File name

    Returns: Data as a raw numpy-data-matrix, the list of the gene names, the list of the experiments
    """

    # Getting data
    expr_data = pd.read_csv(input_path + input_file, index_col=0)

    # Setting columns(=genes) and rows(=experiments)
    gene_names = expr_data.columns
    experiments = expr_data.index.values

    # Converting to numpy
    expr_data = expr_data.to_numpy()
    gene_names = gene_names.to_numpy()

    return expr_data, gene_names, experiments


def main():

    cycle = 0
    termination_condition = False
    G = nx.DiGraph()

    # Read
    gene_data, gene_names, experiments = read_data(arg_input_path, arg_input_file)

    # Mean center & filter out noise: Std threshold, posterior probability (Ludl and Michoel, 2020), random method
    gene_data, gene_names = process_filter(gene_data, gene_names, arg_filt_thres, arg_input_gene_filtering, arg_filter_method)
    print("filtered data shape", gene_data.shape)

    # Graph init, every gene as a node, every experiment: expr val + stats + bayes score
    G = init_nodes(G, gene_data, gene_names, experiments, arg_gamma_prior)

    # Draw exploratory graphs (dendrogram & heatmap) optional display
    dendro(gene_data, arg_figure_path, arg_date_today, arg_dendro)
    heat(gene_data, gene_names, arg_figure_path, arg_date_today, arg_heatmap)

    # Start score-based node combination and edge construction
    node_list = list(G.nodes)
    while not termination_condition:
        cycle += 1

        # Find max scoring pair in graph copy
        node1, node2, pair_b_score, pair_p_score, experiment_values = max_score_pair(G, node_list, experiments, arg_gamma_prior, arg_allow_correction_score)

        # recurrence & redundancy check
        termination_condition = termination_check(termination_condition, G, pair_b_score)
        if termination_condition:
            break

        # Update graph for parent and children of merger
        G, parent_name = update_graph(G, node1, node2, pair_b_score, experiment_values, cycle, arg_plt_space_x, arg_plt_space_mult_y)

        # Add newly created parent to the list of considered genes
        node_list.append(parent_name)

        G, node_list = update_graph_params(G, node_list, node1, node2, parent_name, arg_decay, arg_parent_thres)

    # Test acyclic
    print("Is directed acyclic:", nx.is_directed_acyclic_graph(G))
    print("Graph stats edges + nodes:", G.number_of_edges(), G.number_of_nodes())
    print("GRAPH node Bayesian scores", list(G.nodes(data='bayes_score')))

    # Remove unused genes
    G = prune_graph(G)

    # Change order of genes for heatmap
    sorted_gene_data, sorted_gene_names = sort_graph(G, gene_data, gene_names, arg_sort)

    # Topologically sorted graph into heatmap
    heat(sorted_gene_data, sorted_gene_names, arg_figure_path, arg_date_today, arg_heatmap)

    # Draw matplotlib for DAG
    draw_graph(G, arg_figure_path, arg_date_today, arg_filter_method, arg_plt_label_rota, arg_plt_label_y, arg_graph_type, label=arg_graph_labels)		# 'circular' / 'kamada' / 'spring'

    # Write final graph
    write_result(G, arg_outfile_type, arg_output_file)


if __name__ == '__main__':
    main()
