## Graph Piece

This repo contains the codes for our paper [Graph Piece: Efficiently Generating High-Quality Molecular Graph with Substructures](https://arxiv.org/abs/2106.15098).

The directory **model** contains the complete codes of the graph piece extraction algorithm, our graph piece variational autoencoder (GP-VAE) and the checkpoints / data used in our experiments. If you are interested in training a GP-VAE on your own dataset or running the experiments on GP-VAE, please refer to the instructions provided in that directory.

We have also provided a polished version of the graph piece extraction algorithm decoupled with other codes in the directory **gpe**, which we recommend you to use *if you are only interested in the extracted graph pieces as well as the piecel-level decomposition of molecules*. Please refer to that directory for detailed instructions.

We are still polishing our codes, so feel free to ask about any questions about the codes or problems encountered in running them.