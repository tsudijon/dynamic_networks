''' April 3, 2018
Save the MIT dataset for analysis'''
from __future__ import division
import numpy as np
import os
import sys
from scipy.spatial.distance import squareform
shared_scripts_path = os.path.expanduser('~/projects/dynamic_networks/shared_scripts/')
sys.path.append(shared_scripts_path)
from general_file_fns import save_file
import xlrd


# read in the MIT data;
binsize = '30mins'  # hours
data_path = os.path.expanduser('~/Downloads/mit_data/')
book = xlrd.open_workbook(data_path + 'reality_commons_data_weighted_%s.xlsx' % binsize)
sheet = book.sheet_by_name('reality_commons_data_30_min')
data = np.array([[int(sheet.cell_value(r, c)) for c in range(sheet.ncols)
                if sheet.cell_value(r, c) != ''] for r in range(sheet.nrows)])

edge_wt_list = []
for row in data:
    shape = int(np.sqrt(len(row)))  # size of the whole adjacency matrix
    sq_matrix = np.reshape(row, (shape, shape))  # but diagonal is 1
    np.fill_diagonal(sq_matrix, 0)  # fill diag with 0
    edge_wt_list.append(squareform(sq_matrix))

node_wt_list = []
for i, c in enumerate(edge_wt_list):
    node_wt_list.append(np.zeros(squareform(c).shape[0]).tolist())

to_save = {'edge_wts': edge_wt_list, 'node_wts': node_wt_list}

save_dir = os.path.expanduser('~/Desktop/dyn_networks/analyses/binned_data/bin_%s.p' % binsize)
save_file(to_save, save_dir)
