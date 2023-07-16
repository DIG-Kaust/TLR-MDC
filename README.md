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
situated in [this repository](https://github.com/ecrc/tlrmvm).

## Project structure
This repository is organized as follows:

- **mdctlr**: python library containing routines for tlr-based multidimensional convolution and its use in seismic inverse problem;
- **app**: set of python scripts running the different applications provided by the mdctlr library;
- **script**: set of shell scripts used to run the various applications on standard workstations;
- **script_hpc**: set of shell scripts used to run the various applications on KAUST supercomputer;

## Environments variable

Most of the codes in this repository rely on a number of environment variables. We reccomend to create an `.env` file in the 
root folder. The code will detect this environment file and load the variables inside it.

A sample `.env` file is:

```
FIG_PATH=$HOME/TLRMDCfigures
STORE_PATH=$YOUR_DATASET_PATH
```

- `FIG_PATH`: directory where you want to save your figures.
- `STORE_PATH`: root directory containing the seismic dataset.


## Installation instructions

First install spack and dependencies in `install` folder.
```
./install-gpu.sh
```


Then load dependencies of spack.
```
spack load intel-mkl cmake cuda openmpi
```

Then clone the TLR-MVM library.
```
git clone --recursive git@github.com:ecrc/tlrmvm.git
```

Install TLR-MVM
```
BUILD_CUDA=ON python setup.py build
```

The library will be installed into `build` folder.
go to directory `build/libxxx` and put this build directory 
into your PYTHONPATH.
```
(base) hongy0a@vulture:~/tlrmvm/build/lib.linux-x86_64-3.9$ ls
libtlrmvmcpulib.so  libtlrmvmcudalib.so  pytlrmvm  TLRMVMpy.cpython-39-x86_64-linux-gnu.so
(base) hongy0a@vulture:~/tlrmvm/build/lib.linux-x86_64-3.9$ export PYTHONPATH=$PYTHONPATH:$(pwd)S
```

Then clone TLR-MDC library and put TLR-MDC root directory 
into your PYTHONPATH.
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

You are ready to go!


We also have an installation video on [Youtube](https://www.youtube.com/watch?v=ERRvsPTSn1M).

It will also guide you how to run the application.


## Applications

Available applications:

- generatedataset
- MDC
- Marchenko
- MDDOve3DFull


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


### MDC
To run a single instance of the MDC operator
```
python mdctlr/MDC.py --help
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
mpirun -np 2 python mdctlr/MDC.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16   --ModeValue 8 --OrderType hilbert --debug
```

### Marchenko
To run Marchenko redatuming by inversion for a single virtual point
```
python mdctlr/MarchenkoRedatuming.py --help
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
  --threshold           TLR Threshold
                        TLR Error threshold
  --debug               Debug
```

An example run:
```
mpirun -np 2 python MarchenkoRedatuming.py --AuxFile 3DMarchenko_auxiliary_2.npz --MVMType TLR --TLRType fp16   --ModeValue 8 --OrderType hilbert --debug
```

## Datasets

The codes are based on two different datasets:

- The first one, used to run the MDC, MarchenkoRedatuming, and MDD apps can be downloaded at https://zenodo.org/record/6582600#.Yo-nhJPMKwl. More details about this dataset
can be found in the following publication:

```
@article{ravasi2022,
	title={An open-source framework for the implementation of large-scale integral operators with flexible, modern high-performance computing solutions: 
  Enabling 3D Marchenko imaging by least-squares inversion},
	authors={M. Ravasi, I. Vasconcelos},
	journal={Geophysics},
	year={2021}
}
```

- The second one, used to run the MDDOve3DFull app is currently not available due to data size, contact us if interested in the dataset.