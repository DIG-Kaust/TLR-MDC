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
export STORE_PATH=$STORE_PATH

### Change the following env to your setting.
#export TLRMVMROOT=path_to_tlrmvm_library
#export TLRMDCROOT=path_to_tlrmdc_library
#export PYTHONPATH=$TLRMVMROOT:$TLRMDCROOT

# Run experiments

## Dense
mpirun -np 2 python $TLRMDCROOT/mdctlr/MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType Dense --nfmax 5 --repeat 10 --debug

## TLR-FP16-Normal
mpirun -np 2 python $TLRMDCROOT/mdctlr/MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16 \
  --ModeValue 8 --OrderType normal --nfmax 100 --repeat 10 --debug

## TLR-FP16-Hilbert
mpirun -np 2 python $TLRMDCROOT/mdctlr/MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16 \
  --ModeValue 8 --OrderType hilbert --repeat 10 --debug

## TLR-I8-Hilbert
mpirun -np 2 python $TLRMDCROOT/mdctlr/MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType int8 \
  --ModeValue 8 --OrderType hilbert --repeat 10 --debug
