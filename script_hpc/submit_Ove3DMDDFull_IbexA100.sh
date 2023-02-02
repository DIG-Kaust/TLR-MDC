#!/bin/bash
#SBATCH -n 8
#SBATCH --partition=batch
#SBATCH -J inversion
#SBATCH -o inversion.%J.out
#SBATCH -e inversion.%J.err
#SBATCH --time=00:50:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:8
#SBATCH --reservation=A100


#run the application:

export AIHOME=/ibex/ai/home/ravasim
export FIG_PATH=/home/ravasim/2022/Projects/MDC_TLRMVM/Figs
export STORE_PATH=/ibex/ai/home/ravasim/MDC-TLRMVM
export WORK_ROOT=$STORE_PATH
source $HOME/miniconda3/bin/activate
conda activate mdctlrenv

export TLRMVMROOT=/home/ravasim/2022/Projects/MDC_TLRMVM/tlrmvm-dev/build/lib.linux-x86_64-cpython-39/
export TLRMDCROOT=/home/ravasim/2022/Projects/MDC_TLRMVM/TLR-MDC
export PYTHONPATH=$PYTHONPATH:$TLRMVMROOT:$TLRMDCROOT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TLRMVMROOT

module load gcc/8.2.0
. $HOME/spack/share/spack/setup-env.sh
spack load cmake intel-oneapi-mkl cuda openmpi


# mpirun -np 8 python $TLRMDCROOT/mdctlr/MDDOve3DFull.py --AuxFile 3DMarchenko_auxiliary_2.npz --DataFolder compresseddata_full --M 26040 --N 15930 \
# --MVMType TLR --TLRType fp16 --nb 256 --ModeValue 8 --OrderType normal --nfmax 40 --debug

mpirun -np 8 python $TLRMDCROOT/mdctlr/MDDOve3DFull.py --AuxFile 3DMarchenko_auxiliary_2.npz --DataFolder compresseddata_full --M 26040 --N 15930 \
--MVMType TLR --TLRType fp16 --nb 256 --ModeValue 8 --OrderType hilbert --nfmax 240 --debug
