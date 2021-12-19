# Instructions

## Requirements
networkx >= 2.5
rdkit

## Usage
You can use the following command to set up your graph piece vocabulary:
python mol_bpe.py \
	   --data /path/to/dataset/of/smiles \
	   --vocab_size 500 \
	   --output /path/to/save/the/vocabulary
where vocab_size can be changed according to your need. The Tokenizer defined in the mol_bpe.py will decompose a molecule into a Molecule object (defined in molecule.py). To decompose a molecule, you can run the following commands in python:

```python
from mol_bpe import Tokenizer

smiles = 'COc1cc(C=NNC(=O)c2ccc(O)cc2O)ccc1OCc1ccc(Cl)cc1'
tokenizer = Tokenizer('path/to/the/vocabulary')
mol = tokenizer(smiles)
print('piece level decomposition:')
print(mol)

```
