"""
Wouter Tijs
21/10/2019
University Bergen
"""

import numpy as np
import pandas as pd

######## TEMP ARGS ########
arg_input_gene_filtering = 'processed_yeast_data_AL_1\linear_regression_expr_var4\p2p3_th10pct_gene_interactions.csv'

arg_input_path = '../RawData/'
arg_output_path = '../Results/'

arg_prob_thres = 0.999						# 0.995 is 3709 unique / 0.999 is 1995 unique / 0.9995	is 296 unique / 0.9998 is 39 unique

# Manually selected genes: threshold 0.999
# manual_gene1 = "YLR264W"
# manual_gene2 = "YLR273C"					# YLR326W (183), YLR273C (), YLR257W, YLR252W

manual_gene1 = "YLR271W"
manual_gene2 = "YLR254C"					# YLR252W (118), YLR254C (79), YLR288C (149), YLR264W (142)

# manual_gene1 = "YLR326W"
# manual_gene2 = "YLR326W"					# YLR264W, YLR283W,YLR324W,YLR328W

######## TEMP ARGS ########


def gene_interaction_select(prob_thres, input_file_prob_filt):

	# Importing external file with regulators, targets and posterior probability (Ludl and Michoel, 2020)
	p2p3_selection = pd.read_csv(input_file_prob_filt)

	# Pivot and convert to numpy array
	p2p3_pivot = p2p3_selection.pivot(index="regulator", columns="target", values="weight (posterior probability)")
	gene_prob_sparse = p2p3_pivot.to_numpy()

	# Converting to binary based on the set threshold
	gene_prob_list_binary = np.where(gene_prob_sparse > prob_thres, 1, 0)

	# Matrix multiplication
	interaction_matrix = np.matmul(gene_prob_list_binary, gene_prob_list_binary.T)

	# Set to pandas for sparse matrix evaluation
	gene_names = p2p3_pivot.index.tolist()
	pd_interaction_matrix = pd.DataFrame(interaction_matrix, columns=gene_names, index=gene_names)
	pd_interaction_matrix.to_csv(arg_output_path + "check_matrix_threshold" + str(arg_prob_thres) + "pandas.csv")

	# Write to file for manual check
	np.savetxt(arg_output_path + "check_matrix_threshold" + str(arg_prob_thres) + "_.txt", interaction_matrix, delimiter='\t', fmt='%s')

	# Original matrix to numpy
	gene_prob_list = p2p3_selection.to_numpy()

	# Add to list if meet the threshold
	prob_gene_set = []
	save_to_text = []

	for group in gene_prob_list:
		if group[0] == manual_gene1 or group[0] == manual_gene2:
			if group[2] > prob_thres:
				save_to_text.append(group)
				prob_gene_set += group[0], group[1]

	path_file = arg_output_path + "saved_selected_tar_regu_" + str(arg_prob_thres) + "_" + str(manual_gene1) + "_" + str(manual_gene1) + ".csv"
	np.savetxt(path_file, save_to_text, delimiter=',', fmt='%s')

	# Set for unique genes only
	prob_gene_set = set(prob_gene_set)
	print(len(prob_gene_set))


if __name__ == '__main__':
	gene_interaction_select(arg_prob_thres, arg_input_path + arg_input_gene_filtering)
