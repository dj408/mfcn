"""
Script to process melanoma patient data
files ("scaled_cell_intensities.csv" and
"melanoma_clinical_info_MIBI.csv") into a
list of dictionaries, one per patient manifold,
holding clinical information attributes and
a (sparse) features matrix of scaled cell
sample protein expression levels.

Note: set args in `melanoma/args.py`
before running script.
"""

import sys
sys.path.insert(0, '../')
import argparse
import pandas as pd
import numpy as np
# import torch
import scipy.sparse
from scipy.sparse import (
    csr_matrix,
    diags
)
from collections import Counter
from operator import itemgetter
import pickle
import data_utilities as du
import melanoma.args as a

"""
clargs/args
"""
PROT_EXP_COLS_TO_DROP = ['area', 'x_centroid', 'y_centroid', 'Cell Instance']
PROC_DATA_FILENAME = 'melanoma_manifolds_dictl.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine', default='desktop', type=str, 
                    help='key for the machine in use (default: desktop): see args_template.py')
clargs = parser.parse_args()
args = a.Args(
    MACHINE=clargs.machine,
)


"""
process cell intensity data
"""
# import patients' 'scaled' protein expression data
prot_expression_df = pd.read_csv(
    f'{args.DATA_DIR}/scaled_cell_intensities.csv', 
    header=0, 
    index_col = 0
).astype(pd.SparseDtype("float", 0.))

# remove unneeded columns and set index (54989 rows, 34 cols)
prot_expression_df.drop(PROT_EXP_COLS_TO_DROP, axis=1, inplace=True)
prot_expression_df.index = prot_expression_df.index.str.split('_').str[1]

# extract list of unique patient ids
patient_ids = prot_expression_df.index.unique()
print(f'num. unique patient ids: {len(patient_ids)}')

# convert to scipy.sparse matrix for data transformations
data_arr = csr_matrix(prot_expression_df)

# log2-transform expression values > 0., first adding 1.
# keep 0 values 0 (as if log-transforming after adding 1.)
# pandas sparse should skip operation on 0s
data_arr[data_arr.nonzero()] = np.log2(data_arr[data_arr.nonzero()] + 1.)

# l1-normalize expression values, such that the sum of
# values for each cell sums to 1 (note expression data is > 0)
# (note other papers rescale back up from norm 1. to 1000., not sure why...)
rowsums_diag_matrix = diags(
    1. / data_arr.sum(axis=1).A.ravel(),
    shape=(data_arr.shape[0], data_arr.shape[0]) # (nrow, nrow)
)
data_arr = rowsums_diag_matrix @ data_arr

# inspect
density = data_arr.count_nonzero() / (data_arr.shape[0] * data_arr.shape[1])
print(f'data matrix density: {density:.4f}')
data_arr

# create sparse pandas df
sparse_df = pd.DataFrame.sparse.from_spmatrix(data_arr)
sparse_df.columns = prot_expression_df.columns.tolist()
sparse_df['id'] = prot_expression_df.index
sparse_df.set_index('id', inplace=True)
print(f'sparse_df density: {sparse_df.sparse.density:.4f}')

"""
process clinical info data
"""
# inspect clinical info df
clinical_info = pd.read_csv(
    f'{args.DATA_DIR}/melanoma_clinical_info_MIBI.csv', 
    header=0, 
    index_col=0
)

# remove 'normal' core types
clinical_info.drop(
    clinical_info[clinical_info['Core Type'] == 'normal'].index, 
    inplace=True
)

# cast age at dx to numeric type
clinical_info['AGE_AT_DX'] = pd.to_numeric(
    clinical_info['AGE_AT_DX'], 
    downcast='integer',
    errors='coerce'
)
# new col: gross cancer stage
# first remove any extra notation using a space
clinical_info['DX_STAGE'] = clinical_info['DX_STAGE'] \
    .str.split(' ').str[0]
clinical_info['GROSS_DX_STAGE'] = clinical_info['DX_STAGE'] \
    .map(lambda x: x.rstrip('ABCD'))

# replace 'UNKNOWN' with the majority GROSS_DX_STAGE
gross_dx_stage_ctr = Counter(clinical_info.GROSS_DX_STAGE)
maj_gross_dx_stage = max(
    gross_dx_stage_ctr.items(), 
    key=itemgetter(1)
)[0]
clinical_info['GROSS_DX_STAGE'] = clinical_info['GROSS_DX_STAGE'] \
    .replace('UNKNOWN', maj_gross_dx_stage)

# new col: age at diagnosis tertile
clinical_info['AGE_AT_DX_TERTILE'] = pd.qcut(
    clinical_info['AGE_AT_DX'], 
    3, 
    labels=False
)

# generate id to match cell intensities df
# note in 'scaled_cell_intensities.csv', the 'RXCX' id format
# seems to be R + ['376_1_col'] + 'C' + ['376_1_row'] from
# 'melanoma_clinical_info_MIBI.csv' cols
clinical_info['id'] = 'R' + clinical_info['376_1_col'].astype(str) \
    + 'C' + clinical_info['376_1_row'].astype(str)

# set index to this id
clinical_info.sort_values(by=['id'], inplace=True)
clinical_info.set_index('id', inplace=True)

# filter out patients not in the cell intensities df
patient_sample_cts = Counter(sparse_df.index)
clinical_info = clinical_info[clinical_info.index.isin(patient_sample_cts.keys())]

# inspect
print(len(clinical_info)) # should have 54 rows
clinical_info.head(10)

"""
collect cell and clinical data into records
"""
# CR = complete response, PR = partial response, SD = stable disease, 
# PD = progressive disease, and NE = inevaluable.
# response_multi_dict = {'PD': 0, 'SD': 1, 'PR': 2, 'CR': 3}

data_records = [None] * len(clinical_info)
for i, pt_id in enumerate(clinical_info.index):
    ci_row = clinical_info.loc[pt_id]
    response_binary = 1 if 'y' in ci_row['RESPONSE'].lower() else 0
    # response_multi = response_multi_dict[ci_row['BEST_RESPONSE_BY_SCAN']]
    response_multi = ci_row['BEST_RESPONSE_BY_SCAN']

    cell_intensities = sparse_df.loc[pt_id] \
        .sparse.to_coo()
    
    data_records[i] = {
        'pt_id': pt_id,
        'age_at_dx': int(ci_row['AGE_AT_DX']),
        'age_at_dx_tertile': int(ci_row['AGE_AT_DX_TERTILE']),
        'sex': ci_row['SEX'],
        'race': ci_row['RACE'],
        'dx_stage': ci_row['DX_STAGE'],
        'gross_dx_stage': ci_row['GROSS_DX_STAGE'],
        'response_binary': response_binary,
        'response_multi': response_multi,
        'cell_intensities': cell_intensities
    }

"""
pickle records
"""
save_path = f'{args.DATA_DIR}/{PROC_DATA_FILENAME}'
with open(save_path, 'wb') as f:
    pickle.dump(data_records, f, protocol=pickle.HIGHEST_PROTOCOL)




