

# Instructions

## Requirements
 - pytorch_geometric  # Maybe you need to refer to its documentation for installation
 - pytorch_lightning
 - rdkit

## Usage

Please add the root directory of our codes to the environment variable PYTHONPATH before running any scripts.

For example, if the path to our codes is `~/gp-vae`, then the following command is needed:

```bash
export PYTHONPATH=~/gp-vae/model:$PYTHONPATH
```

### Graph Piece Extraction
```bash
python data/mol_bpe.py \
    --data /path/to/your/dataset \
    --output /path/to/vocabfile \
    --vocab_size size_of_the_vocabulary
```

We have also provided a polished version of the graph piece extraction algorithm in the directory **gpe**, which we recommend you to use if you are only interested in the piecel-level decomposition of molecules.

### Training

We have provided trained checkpoints in zinc_exps/ckpt/300/epoch5.ckpt for a graph piece variational autoencoder with a vocabulary of 300 graph pieces and zinc_exps/ckpt/500/epoch5.ckpt for one with a vocabulary of 500 graph pieces. These two checkpoints are used for the property optimization task and constrained property optimization task, respectively.
You can also train your model as follows. The data will be automatically preprocessed on the first access and produce a processed_data.pkl in the same directory of the data. Afterwards the processed data will be automatically loaded instead of reprocessed.

```bash
python train.py \
	--train_set zinc_exps/data/train_zinc250k/train.txt \
	--valid_set zinc_exps/data/valid_zinc250k/valid.txt \
	--test_set zinc_exps/data/test_zinc250k/test.txt \
	--vocab zinc_exps/ckpt/300/zinc_bpe_300.txt \
	--batch_size 32 \
	--shuffle \
	--alpha 0.1 \
	--beta 0 \
	--max_beta 0.01 \
	--step_beta 0.002 \
	--kl_anneal_iter 1000 \
	--kl_warmup 0 \
	--lr 1e-3 \
	--save_dir zinc_exps/ckpt/yours \
	--grad_clip 10.0 \
	--epochs 6 \
	--gpus 0 \
	--model vae_piece \
	--props qed logp \
	--latent_dim 56 \
	--node_hidden_dim 300 \
	--graph_embedding_dim 400 \
	--patience 3
```

### Distribution Learning
```bash
python guacamol_exps/distribution_learning.py \
  --model vae_piece \
  --ckpt /path/to/checkpoint \
  --gpu 0 \
  --output_dir results \
  --dist_file /path/to/train/set \
```

### Property Optimization
You can generate molecules with optimized properties as follows. If you want to add multi-objective constraints, you can use a comma to split the properties (e.g. qed,logp). We recommend the checkpoint in zinc_exps/ckpt/300 for this task.
```bash
python generate.py --eval \
    	 --ckpt /path/to/checkpoint \
         --props qed \
         --n_samples 10000 \
         --output_path qed.smi \
         --optimize_method direct \
         --model vae_piece \
         --lr 0.1 \
         --max_iter 100 \
         --patience 3 \
         --target 2 \
         --cpus 8 \
```

### Constrained Property Optimization
We copy the 800 molecules with the lowest Penalized logP in the test set from the offical codes of JTVAE, just as the way GCPN does. We recommend the checkpoint in zinc_exps/ckpt/500 for this task.
```bash
python generate.py \
    --ckpt /path/to/checkpoint \
    --props logp \
    --n_samples 800 \
    --train_set zinc_exps/data/train_zinc250k/train.txt \
    --output_path cons_res \
    --optimize_method direct \
    --model vae_piece \
    --lr 0.1 \
    --max_iter 80 \
    --constraint_optim \
    --zinc800_logp zinc_exps/data/jtvae_zinc800_logp.smi \
    --cpus 8 \
    --gpus 0
```
