#!/bin/bash

# MDC EXAMPLES
##############

# Set-up environment
. $HOME/spack/share/spack/setup-env.sh
spack load intel-oneapi-mkl@2022.0.2
spack load cuda@11.5.1
spack load openmpi@4.1.3
spack load cmake@3.21.0
conda activate mdctlr
export FIG_PATH=$HOME/figs
export STORE_PATH=/home/ravasim/Documents/Data/Overtrust3D/ #$STORE_PATH

### Change the following env to your setting.
#export TLRMVMROOT=path_to_tlrmvm_library
#export TLRMDCROOT=path_to_tlrmdc_library
#export PYTHONPATH=$TLRMVMROOT:$TLRMDCROOT

# Run experiments

## TLR-FP16-Normal (works with tlrmvm yuxi/dev)
mpirun -np 2 python $TLRMDCROOT/mdctlr/MDDOve3D.py --AuxFile 3DMarchenko_auxiliary_2.npz --M 6510 --N 15930 --MVMType TLR --TLRType fp16  \
 --nb 128 --ModeValue 4 --OrderType normal --nfmax 40 --debug