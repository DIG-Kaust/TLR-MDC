#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=batch
#SBATCH -J mdc
#SBATCH -o mdc.%J.out
#SBATCH -e mdc.%J.err
#SBATCH --time=01:00:00
#SBATCH --mem=300G
#SBATCH --gpus=v100:4


## MDC of Syncline 3D model, to be run as: sbatch submit_MDC_IbexV100.sh

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


# run the application (Dense) - NOT WORKING YET:
mpirun -np 4 python $TLRMDCROOT/app/MDC.py --AuxFile 3DMarchenko_aux.npz --MVMType Dense --nfmax 50 --repeat 10 --debug

# run the application (TLR-FP16-Normal):
mpirun -np 4 python $TLRMDCROOT/app/MDC.py --AuxFile 3DMarchenko_aux.npz --MVMType TLR --TLRType fp16 \
  --ModeValue 8 --OrderType normal --repeat 10 --debug
  
# run the application (TLR-FP16-Hilbert):
mpirun -np 4 python $TLRMDCROOT/app/MDC.py --AuxFile 3DMarchenko_aux.npz --MVMType TLR --TLRType fp16 \
  --ModeValue 8 --OrderType hilbert --repeat 10 --debug