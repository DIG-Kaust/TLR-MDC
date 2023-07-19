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


## Installation instructions

TO BE UPDATED!

We also have an installation video on [Youtube](https://www.youtube.com/watch?v=ERRvsPTSn1M) - OUTDATED!

It will also guide you how to run the application.


## Applications and inputs

Available applications:

- GenerateDataset
- MDC
- Marchenko
- MDD
- MDDOve3DFull


### GenerateDataset
To create a TLR compressed version of a dataset:

```
python app/GenerateDataset.py --help
```

will give you

```
usage: GenerateDataset.py [-h] [--nb NB] [--error_threshold ERROR_THRESHOLD] [--reordering REORDERING] [--freqlist FREQLIST] [--rankmodule RANKMODULE] [--nrx NRX] [--nry NRY] [--foldername FOLDERNAME] [--prefix PREFIX] [--suffix SUFFIX] [--format FORMAT] [--matname MATNAME]

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
  --nrx NRX             number of receivers along the x axis
  --nry NRY             number of receivers along the y axis
  --foldername FOLDERNAME
                        foldername where input data are stored
  --prefix PREFIX       prefix of filenames
  --suffix SUFFIX       suffix of filenames
  --format FORMAT       format of file (mat or zarr)
  --matname MATNAME     name of variable in matfile


```

An example run:
```
python app/GenerateDataset.py \
    --nb=256 --error_threshold=0.001 --reordering=normal --rankmodule=8 \
    --freqlist=0,1,
```


### MDC
To run a single instance of the MDC operator
```
python app/MDC.py --help
```

will give you

```
usage: MDC.py [-h] [--AuxFile AUXFILE] [--MVMType MVMTYPE] [--TLRType TLRTYPE] [--bandlen BANDLEN] [--nfmax NFMAX] [--wavfreq WAVFREQ] [--OrderType ORDERTYPE] [--ModeValue MODEVALUE] [--M M] [--N N] [--nb NB] [--threshold THRESHOLD] [--repeat REPEAT] [--debug]

3D Multi-Dimensional Convolution with TLR-MDC and matrix reordering

optional arguments:
  -h, --help            show this help message and exit
  --AuxFile AUXFILE     File with Auxiliary information for MDC
  --MVMType MVMTYPE     Type of MVM: Dense, TLR
  --TLRType TLRTYPE     TLR Precision: fp32, fp16, fp16int8, int8
  --bandlen BANDLEN     TLR Band length
  --nfmax NFMAX         TLR Number of frequencies
  --wavfreq WAVFREQ     Ricker wavelet peak frequency used to convolve the input
  --OrderType ORDERTYPE
                        Matrix reordering method: normal, l1, hilbert
  --ModeValue MODEVALUE
                        Rank mode
  --M M                 Number of sources/rows in seismic frequency data
  --N N                 Number of receivers/columns in seismic frequency data
  --nb NB               TLR Tile size
  --threshold THRESHOLD
                        TLR Error threshold
  --repeat REPEAT       Number of repeated MDC computation for statistics
  --debug               Debug

```

An example run:
```
mpirun -np 4 python app/MDC.py --AuxFile 3DMarchenko_aux.npz --MVMType TLR --TLRType fp16 \
  --ModeValue 8 --OrderType normal --repeat 10 --debug
 ```

The Auxiliary .npz file must contain the following variables:

- :card_index: ``x``: x-axis of subsurface model
- :card_index: ``y``: y-axis of subsurface model
- :card_index: ``z``: z-axis of subsurface model
- :card_index: ``t``: t-axis of data
- :card_index: ``srcs``: sources coordinates (array of shape nsrc x 3 ordered as x,y,z)
- :card_index: ``recs``: receivers coordinates (array of shape nsrc x 3 ordered as x,y,z)
- :card_index: ``vs``: chosen virtual source (array of shape 3 x 1 ordered as x,y,z)
- :card_index: ``G0``: reference wavefield to compare  (array of shape nt x nrecs ordered as x,y,z)


### Marchenko
To run Marchenko redatuming by inversion for a single virtual point
```
python app/MarchenkoRedatuming.py --help
```

will give you

```
usage: MarchenkoRedatuming.py [-h] [--AuxFile AUXFILE] [--MVMType MVMTYPE] [--TLRType TLRTYPE] [--bandlen BANDLEN] [--nfmax NFMAX] [--OrderType ORDERTYPE] [--ModeValue MODEVALUE] [--M M] [--N N] [--nb NB] [--threshold THRESHOLD] [--debug]

3D Marchenko Redatuming with TLR-MDC and matrix reordering

optional arguments:
  -h, --help            show this help message and exit
  --AuxFile AUXFILE     File with Auxiliar information for Mck redatuming
  --MVMType MVMTYPE     Type of MVM: Dense, TLR
  --TLRType TLRTYPE     TLR Precision: fp32, fp16, int8
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
mpirun -np 4 python app/MarchenkoRedatuming.py --AuxFile 3DMarchenko_aux.npz --MVMType TLR --TLRType fp16 \
  --ModeValue 8 --OrderType normal --debug

```


The Auxiliary .npz file must contain the same variables of the MDC app, plus:

- :card_index: ``G``: full wavefield to compare  (array of shape nt x nrecs ordered as x,y,z)



### MDD
To run MDD of Marchenko redatumed fields for a single virtual point
```
python app/MDD.py --help
```

will give you

```
usage: MDD.py [-h] [--AuxFile AUXFILE] [--MVMType MVMTYPE] [--TLRType TLRTYPE] [--bandlen BANDLEN] [--nfmax NFMAX] [--OrderType ORDERTYPE] [--ModeValue MODEVALUE] [--M M] [--N N] [--ivsinv IVSINV] [--nb NB] [--threshold THRESHOLD] [--debug]

3D Multi-Dimensional Deconvolution with TLR-MDC and matrix reordering

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
  --ivsinv IVSINV       Index of virtual source to invert for
  --nb NB               TLR Tile size
  --threshold THRESHOLD
                        TLR Error threshold
  --debug               Debug
```

An example run:
```
mpirun -np 4 python $TLRMDCROOT/app/MDD.py --AuxFile 3DMDD_aux.npz --M 9801 --N 2911 --MVMType TLR --TLRType fp16 \
  --nb 128 --ModeValue 4 --OrderType normal --nfmax 150 --ivsinv 880 --debug

```


The Auxiliary .npz file must contain the following variables:

- :card_index: ``srcs``: sources coordinates (array of shape nsrc x 3 ordered as x,y,z)
- :card_index: ``vsz``: depth of array of grid of virtual receveirs (scalar)
- :card_index: ``vsx``: x-axis of grid of virtual receivers
- :card_index: ``vsy``: y-axis of grid of virtual receivers
- :card_index: ``t``: t-axis of data


### MDDOve3DFull
To run MDD of Up/Down separated fields for a single virtual point (currently implemented for Overthrust model, but code should be
generic and applicable to any other data provided it is organized according to our requirements - see below)
```
python app/MDDOve3DFull.py --help
```

will give you

```
usage: MDDOve3DFull.py [-h] [--AuxFile AUXFILE] [--DataFolder DATAFOLDER] [--PupFolder PUPFOLDER] [--FigFolder FIGFOLDER] [--MVMType MVMTYPE] [--TLRType TLRTYPE] [--bandlen BANDLEN] [--nfmax NFMAX] [--OrderType ORDERTYPE] [--PHilbertSrc PHILBERTSRC] [--PHilbertRec PHILBERTREC] [--ModeValue MODEVALUE] [--M M] [--N N] [--nb NB] [--threshold THRESHOLD] [--vs VS] [--niter NITER]
                       [--damp DAMP] [--debug]

3D Multi-Dimensional Deconvolution with TLR-MDC and matrix reordering

optional arguments:
  -h, --help            show this help message and exit
  --AuxFile AUXFILE     File with Auxiliary information for MDD
  --DataFolder DATAFOLDER
                        Folder containing compressed data
  --PupFolder PUPFOLDER
                        Folder containing pup data
  --FigFolder FIGFOLDER
                        Prefix of folder containing figures
  --MVMType MVMTYPE     Type of MVM: Dense, TLR
  --TLRType TLRTYPE     TLR Precision: fp32, fp16, fp16int8, int8
  --bandlen BANDLEN     TLR Band length
  --nfmax NFMAX         TLR Number of frequencies
  --OrderType ORDERTYPE
                        Matrix reordering method: normal, l1, hilbert
  --PHilbertSrc PHILBERTSRC
                        Hilbert size of source axis (2^p)
  --PHilbertRec PHILBERTREC
                        Hilbert size of receiver axis (2^p)
  --ModeValue MODEVALUE
                        Rank mode
  --M M                 Number of sources/rows in seismic frequency data
  --N N                 Number of receivers/columns in seismic frequency data
  --nb NB               TLR Tile size
  --threshold THRESHOLD
                        TLR Error threshold
  --vs VS               Virtual source
  --niter NITER         Number of iterations of MDD
  --damp DAMP           Damping of MDD
  --debug               Debug
```

An example run:
```
mpirun -np 4 python $TLRMDCROOT/app/MDDOve3DFull.py --AuxFile MDDOve3D_aux.npz --DataFolder compresseddata_full \
--M 26040 --N 15930 --MVMType TLR --TLRType fp32 \
--nb 256 --threshold 0.001 --ModeValue 8 --OrderType hilbert --PHilbertSrc 12 --PHilbertRec 12 --nfmax 200 --vs 9115 --debug

```


The Auxiliary .npz file must contain the following variables:

- :card_index: ``ny``: number of samples of y-axis of subsurface model
- :card_index: ``nx``: number of samples of x-axis of subsurface model
- :card_index: ``nz``: number of samples of z-axis of subsurface model
- :card_index: ``dy``: sampling of y-axis of subsurface model
- :card_index: ``dx``: sampling of x-axis of subsurface model
- :card_index: ``dz``: sampling of z-axis of subsurface model
- :card_index: ``osy``: origin of source grid over y-axis (the same gap between the origin of the reference system and the origin of the source array, will also be added at the end of the grid)
- :card_index: ``dsy``: sampling of source grid over y-axis 
- :card_index: ``osx``: origin of source grid over x-axis 
- :card_index: ``dsx``: sampling of source grid over x-axis 
- :card_index: ``ory``: origin of receiver grid over y-axis
- :card_index: ``dry``: sampling of receiver grid over y-axis 
- :card_index: ``nry``: number of samples of receiver grid over y-axis 
- :card_index: ``orx``: origin of receiver grid over x-axis 
- :card_index: ``drx``: sampling of receiver grid over x-axis 
- :card_index: ``nrx``: number of samples of receiver grid over x-axis 
- :card_index: ``t``: t-axis of data



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