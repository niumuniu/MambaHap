# MambaHap：A Mamba-Based Selective State-Space Model for Accurate Haplotype Assembly

## About

MambaHap is a haplotype assembly tool that integrates the Mamba deep learning model with spectral clustering to efficiently capture long-range dependencies and reconstruct complete haplotype sequences.

## Dependencies

Python >= 3.10

PyTorch >= 2.6

Mamba

Numpy

C++

Samtools



## Assumed directory structure



The `data` folder stores all the files required for data generation in the experiments of this project.

## Input

 Before executing the MambaHap pipeline, the `SNV_matrix.txt` file must be generated through preprocessing, which serves as the input to the model.

## Output

The best results eventually obtained by the MambaHap pipeline are stored in the following files:

`MambaHap.npz`:**.npz** file storing the reconstructed haplotype(rec_hap), the read attribute (rec_hap_origin), and the ground-truth haplotype(true_hap) if applicable.

`MambaHap_model`:**State_dict** storing the convolutional(embedAE)and mamba coding layers(mamba) in MambaHap.

## Usage

Run **main_Mambahap.py** :

```
python main_Mambahap.py -f soltub_region1 -p 4 -a 5 -g 3
```

Run **main_sparse.py** ：

```
python mian_sparse.py -f chr21 -p 2 -a 1 -g 3
```

Where `-f` specifies the prefix of the required files, `-r` indicates the reference genome in FASTA format, `-p` denotes the ploidy of the data, `-c` represents the sequencing depth of the reads, `-a` sets the number of experimental runs, and `-g` specifies the GPU ID used to run MambaHap.
