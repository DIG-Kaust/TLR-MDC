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

### Change the following env to your setting.
#export FIG_PATH=path_to_figure_folder
#export STORE_PATH=path_to_data_folder
#export TLRMVMROOT=path_to_tlrmvm_library
#export TLRMDCROOT=path_to_tlrmdc_library
#export PYTHONPATH=$TLRMVMROOT:$TLRMDCROOT

# Run experiments

## Normal ordering dataset
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=41 --nry=71 \
    --foldername=Gplus_freqslices --prefix=Gplus_freqslice --suffix=_sub1 --matname=Gplusfreq \
    --freqlist=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4 --nrx=41 --nry=71 \
    --foldername=Gplus_freqslices --prefix=Gplus_freqslice --suffix=_sub1 --matname=Gplusfreq \
    --freqlist=50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
python $TLRMDCROOT/mdctlr/tlrmvm/generatedataset.py \
    --nb=128 --error_threshold=0.001 --reordering=normal --rankmodule=4  --nrx=41 --nry=71 \
    --foldername=Gplus_freqslices --prefix=Gplus_freqslice --suffix=_sub1 --matname=Gplusfreq \
    --freqlist=100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149
