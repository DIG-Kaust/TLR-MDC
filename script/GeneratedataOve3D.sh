#!/bin/bash

# MARCHENKO REDATUMING EXAMPLES
###############################

# Set-up environment
. $HOME/spack/share/spack/setup-env.sh
spack load intel-oneapi-mkl@2022.0.2
spack load cuda@11.5.1
spack load openmpi@4.1.3
spack load cmake@3.21.0
conda activate mdctlr
export FIG_PATH=/home/ravasim/Documents/2022/Projects/TLR-MDC/figresults/
export STORE_PATH=/home/ravasim/Documents/Data/Overtrust3D/

### Change the following env to your setting.
#export TLRMVMROOT=path_to_tlrmvm_library
#export TLRMDCROOT=path_to_tlrmdc_library
#export PYTHONPATH=$TLRMVMROOT:$TLRMDCROOT

# Run experiments

## TLR-FP16-Normal

# test
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=177 --nry=90 \
    --foldername=Data --prefix=PDOWN --suffix='' --format=zarr \
    --freqlist=20

python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=177 --nry=90 \
    --foldername=Data --prefix=PDOWN --suffix='' --format=zarr \
    --freqlist=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=177 --nry=90 \
    --foldername=Data --prefix=PDOWN --suffix='' --format=zarr \
    --freqlist=50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=177 --nry=90 \
    --foldername=Data --prefix=PDOWN --suffix='' --format=zarr \
    --freqlist=100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=177 --nry=90 \
    --foldername=Data --prefix=PDOWN --suffix='' --format=zarr \
    --freqlist=150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=177 --nry=90 \
    --foldername=Data --prefix=PDOWN --suffix='' --format=zarr \
    --freqlist=200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259