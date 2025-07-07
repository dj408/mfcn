#!/bin/bash
#SBATCH -J melanoma         # job name
#SBATCH -p gpu-l40          # queue (partition)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (no MPI)
#SBATCH --gres=gpu:1        # request a gpu
#SBATCH -c 16               # cpus per task
#SBATCH -t 0-00:00:30       # run time (d-hh:mm:ss)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=YOUR_OUTPUT_DIR/mfcn/job_outputs/%j

# execute bashrc stuff
. ~/.bashrc

# useful env variables
export root="YOUR_ROOT_DIR"
export OMP_NUM_THREADS=1
export TQDM_DISABLE=1

# if using conda
conda activate torch-env

# if using python venv:
# module load python/3.9.7
# module load cudnn8.7-cuda11/8.7.0.84
# source ${root}/mfcn/.venv/bin/activate

# ensure own files/modules can be imported in other files
export PYTHONPATH="${root}/mfcn/code":$PYTHONPATH

# run script
python3 "${root}/mfcn/code/train_scripts/experiments_cv.py" \
--machine borah \
--mcn_spectral \
--spectral_c 0.5 \
--mcn_within_filter_chan_out 32,16 \
--dataset "melanoma" \
--n_folds 10 \
--n_epochs 1000 \
--burn_in 32 \
--patience 32 \
--learn_rate 0.005 \
--batch_size 8 \
--verbosity 0
