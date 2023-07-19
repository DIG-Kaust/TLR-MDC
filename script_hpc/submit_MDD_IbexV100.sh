#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=batch
#SBATCH -J mdd
#SBATCH -o mdd.%J.out
#SBATCH -e mdd.%J.err
#SBATCH --time=01:00:00
#SBATCH --mem=300G
#SBATCH --gpus=v100:4


## MDD of Syncline 3D model, to be run as: sbatch submit_MDD_IbexV100.sh

# load environment
module load intel/2020 gcc/10.2.0 openmpi/4.0.3-cuda11.2.2 cmake/3.24.2/gnu-8.2.0 cuda/11.2.2
source /home/ravasim/miniconda3/bin/activate mdctlrenv
export TLRMDCROOT=/home/ravasim/2022/Projects/MDC_TLRMVM_v2/TLR-MDC
export TLRMVMROOT=/home/ravasim/2022/Projects/MDC_TLRMVM_v2/tlrmvm-dev/build/install/lib
export PYTLRROOT=/home/ravasim/2022/Projects/MDC_TLRMVM_v2/tlrmvm-dev/build/install/python
export PYTHONPATH=$PYTHONPATH:$TLRMDCROOT:$TLRMVMROOT:$PYTLRROOT
export PYTHONDONTWRITEBYTECODE=1 # disable generation of __pycache__ folder
export STORE_PATH=/ibex/ai/home/ravasim/ravasim_OLDscratch/MDC-TLRMVM/
export FIG_PATH=/home/ravasim/2022/Projects/MDC_TLRMVM_v2/Figs

# run the application (Dense):
mpirun -np 4 python $TLRMDCROOT/app/MDD.py --AuxFile 3DMDD_aux.npz --M 9801 --N 2911 --MVMType Dense \
  --nfmax 150 --ivsinv 880 --debug

# run the application (TLR-FP16-Normal):
mpirun -np 4 python $TLRMDCROOT/app/MDD.py --AuxFile 3DMDD_aux.npz --M 9801 --N 2911 --MVMType TLR --TLRType fp16 \
  --nb 128 --ModeValue 4 --OrderType normal --nfmax 150 --ivsinv 880 --debug
