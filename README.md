# TLR-MDC 
Tile-Low Rank Multi-Dimensional Convolution: fast MDC modelling and inversion for seismic applications

Tile-Low Rank Multi-Dimensional Convolution (TLR-MDC) represents one of the most expensive operators in seismic processing. 
This is especially the case for large 3D seismic datasets, as it requires access to the entire data in the form of a 3D tensor 
(freq-sources-recs) for the application of a batched complex-valued matrix-vector multiplication operation.

This repository contains an extemely efficient implementation of MDC by leveraging data sparsity encountered in seismic 
frequency-domain data. More precisely, each seismic frequency slice is first tiled, compressed by means of singular value 
decomposition and its bases are stored on disk (TLR compression). Subsequently,  the MDC operator is applied by operating 
directly with the compressed bases using a custom-made TLR-MVM kernel.

This repository provides Python codes to run a variety of seismic algorithms leveraging the C++/CUDA implementation of TLR-MVM 
situated in [this repository](TLR-MVM REPO).

## Project structure
This repository is organized as follows:

- **mdctlr**: python library containing routines for tlr-based multidimensional convolution and its use in seismic inverse problem
- **apps**: set of python scripts implementing various seismic applications (Marchenko redatuming, Marchenko demultiple, MDD)
- **scripts**: set of shell scripts used to run the various applications on standard workstations and KAUST supercomputer
- **install**: set of shell scripts used in the installation process

## Environments variable

Most of the codes in this repository rely on a number of environment variables. We reccomend to create an `.env` file in the 
root folder. The code will detect this environment file and load the variables inside it.

A sample `.env` file is:

```
FIG_PATH=/home/$USER//TLR-MDC_figures
STORE_PATH=/home/$USER/Data
PROJECT_ROOT=/home/$USER/TLR-MDC
PYTHONPATH=/home/$USER/TLR-MDC/:/home/$USER/TLR-MDC/tlrmvm-dev/build/lib.linux-x86_64-3.9:
```

where:
-`FIG_PATH`: directory where you want to save your figures.
-`STORE_PATH`: root directory containing the seismic dataset.
-`PROJECT_ROOT`: directory of this repo.
-`PYTHONPATH`: directory of the TLR-MDC python library of this repo (and the tlrmvm build directory).


## Installation instructions

The installation process involves 2 separate step:

### Installation of tlrmvm

...

### Installation of mdctlr

...

## Applications

Available applications:

- generatedataset
- mdctlr.caldatasize 
- MDC
- Marchenko

### generatedataset
To create a TLR compressed version of the original dataset:

```
python mdctlr/tlrmvm/generatedataset.py --help
```

will give you

```
usage: generatedataset.py [-h] [--nb NB] [--error_threshold ERROR_THRESHOLD] [--reordering REORDERING] [--freqlist FREQLIST]
                          [--rankmodule RANKMODULE]

optional arguments:
  -h, --help            show this help message and exit
  --nb NB               nb
  --error_threshold ERROR_THRESHOLD
                        error threshold
  --reordering REORDERING
                        geometry reordering type: hilbert, normal
  --freqlist FREQLIST   processing freqlist
  --rankmodule RANKMODULE
                        all rank in the matrix dividable by certain value

```

An example run:
```
python $(pwd)/mdctlr/tlrmvm/generatedataset.py \
    --nb=256 --error_threshold=0.001 --reordering=normal --rankmodule=8 \
    --freqlist=0,1,
```

See more at `scripts/generateordering`.


### mdctlr.caldatasize 
To obtain statistics about a give TLR-compressed dataset. 
```
mdctlr.caldatasize --help
```

will give you

```
usage: mdctlr.caldatasize [-h] [--storepath STOREPATH] [--order ORDER] [--bandlength BANDLENGTH] [--outtype OUTTYPE] [--intype INTYPE]

optional arguments:
  -h, --help            show this help message and exit
  --storepath STOREPATH
                        your dataset store path.
  --order ORDER         geometry order type: hilbert, normal
  --bandlength BANDLENGTH
                        band length of inner band
  --outtype OUTTYPE     precision of inner band
  --intype INTYPE       precison of inner band
```

An example run:
```
mpirun -np 2 python MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16   --ModeValue 8 --OrderType hilbert --debug
```
will give you 
```
Your data path:  /datawaha/ecrc/hongy0a/seismic
ordering method:  normal
band length:  0
full outtype  fp32
size is  32.391774208  GB
```

### MDC
To run a single instance of the MDC operator
```
python apps/MDC.py --help
```

will give you

```
usage: MDC.py [-h] [--AuxFile AUXFILE] [--MVMType MVMTYPE] [--TLRType TLRTYPE] [--bandlen BANDLEN] [--nfmax NFMAX]
              [--OrderType ORDERTYPE] [--ModeValue MODEVALUE] [--M M] [--N N] [--nb NB] [--threshold THRESHOLD] [--debug]

3D Multi-Dimensional Convolution with TLR-MDC and matrix reordering

optional arguments:
  -h, --help            show this help message and exit
  --AuxFile AUXFILE     File with Auxiliar information for Mck redatuming
  --MVMType MVMTYPE     Type of MVM: Dense, TLR
  --TLRType TLRTYPE     TLR Precision: fp32, fp16, fp16int8, int8
  --bandlen BANDLEN     TLR Band length
  --nfmax NFMAX         TLR Number of frequencies
  --OrderType ORDERTYPE
                        Matrix reordering method: normal, l1, hilbert
  --ModeValue MODEVALUE
                        Rank mode
  --M M                 Number of sources/rows in seismic frequency data
  --N N                 Number of receivers/columns in seismic frequency data
  --nb NB               TLR Tile size
  --threshold THRESHOLD
                        TLR Error threshold
  --debug               Debug
```

An example run:
```
mpirun -np 2 python MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16   --ModeValue 8 --OrderType hilbert --debug
```

### Marchenko
To run Marchenko redatuming by inversion for a single virtual point
```
python MarchenkoRedatuming.py --help
```

will give you

```
usage: MarchenkoRedatuming.py [-h] [--AuxFile AUXFILE] [--MVMType MVMTYPE] [--TLRType TLRTYPE] [--bandlen BANDLEN]
                              [--nfmax NFMAX] [--OrderType ORDERTYPE] [--ModeValue MODEVALUE] [--M M] [--N N] [--nb NB]
                              [--threshold THRESHOLD] [--debug]

3D Marchenko Redatuming with TLR-MDC and matrix reordering

optional arguments:
  -h, --help            show this help message and exit
  --AuxFile AUXFILE     File with Auxiliar information for Mck redatuming
  --MVMType MVMTYPE     Type of MVM: Dense, TLR
  --TLRType TLRTYPE     TLR Precision: fp32, fp16, fp16int8, int8
  --bandlen BANDLEN     TLR Band length
  --nfmax NFMAX         TLR Number of frequencies
  --OrderType ORDERTYPE
                        Matrix reordering method: normal, l1, hilbert
  --ModeValue MODEVALUE
                        Rank mode
  --M M                 Number of sources/rows in seismic frequency data
  --N N                 Number of receivers/columns in seismic frequency data
  --nb NB               TLR Tile size
  --threshold THRESHOLD
                        TLR Error threshold
  --debug               Debug
```

An example run:
```
mpirun -np 2 python MarchenkoRedatuming.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16   --ModeValue 8 --OrderType hilbert --debug
```

## Dataset

The codes are based on MDC dataset. The dataset contains 300 frequency matrices which are located at `${STORE_PATH}/Mck_freqslices`. Each matrix name is `Mck_freqslice{freqid}_sub1.mat`. Frequency id ranges from 0 to 299. The size of frequency matrix is `9801 x 9801`. Below is an example to load origin matrix 

```
from scipy.io import loadmat
A = loadmat("Mck_freqslice100_sub1.mat")['Rfreq']
```

