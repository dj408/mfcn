"""
From either (a) CSV files of melanoma patient
data, or (b) a pickled data_dictl (records) in
melanoma.args.RAW_DATA_FILENAME (e.g.
'melanoma_manifolds_dictl.pkl'), generate 
train/valid/test pytorch-geometric datasets
for the melanoma graph classification task.
"""

import sys
sys.path.insert(0, '../')
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

import data_utilities as du
import dataset_creation as dc
import melanoma.args as a


"""
clargs/args
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine', default='desktop', type=str, 
                    help='key for the machine in use (default: desktop): see args_template.py')
parser.add_argument('--from_pickle', default=False, action='store_true')
clargs = parser.parse_args()
args = a.Args(
    MACHINE=clargs.machine,
)

"""
load and combine datasets
"""
if clargs.from_pickle:
    # load from pickled list of patient data records (dicts list)
    with open(f"{args.DATA_DIR}/{args.RAW_DATA_FILENAME}", "rb") as f:
        data_dictl = pickle.load(f)
else:
    # load from CSVs (default)
    df_pt_info = pd.read_csv(f"{args.DATA_DIR}/patient_info.csv")
    df_protein_exp = pd.read_csv(f"{args.DATA_DIR}/cell_protein_data.csv")
    df_spatial = pd.read_csv(f"{args.DATA_DIR}/cell_spatial_data.csv")
    data_dictl = df_pt_info.to_dict('records')
    
    for rec in data_dictl:
        prot_exp_df = df_protein_exp[df_protein_exp['id'] == rec['id']].copy()
        prot_exp_df.drop('id', axis=1, inplace=True)
        rec['cell_intensities'] = coo_matrix(prot_exp_df)
        rec['spatial_data'] = df_spatial[df_spatial['id'] == rec['id']].copy()

# create manifold wavelet dataset, holding spectral objects if 
# args.SAVE_SPECTRAL_OBJS is True (so they're only calculated once)
data_dictl = dc.create_manifold_wavelet_dataset(
    args,
    data_dictl,
    pickle_dictl_out=False
)

# convert to pytorch-geometric Data objects and save
extra_attribs = {
    'id': str,
    'age_at_dx_tertile': int,
    'dx_stage': str,
    'gross_dx_stage': str,
    'response_binary': int,
    'response_multi': str,
}
if args.SAVE_SPECTRAL_OBJS:
    extra_attribs['L_eigenvecs'] = float
    extra_attribs['L_eigenvals'] = float
dc.pickle_as_pytorch_geo_Data(args, data_dictl, extra_attribs)

