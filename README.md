<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

This is the official repository of MuTarget paper.

# To Do

- [ ] Prepare install.sh file
- [ ] Prepare Readme file
- [ ] Refactor prediction code

# Usage

## Installation

Clone the repository and navigate into the directory:

```
git clone [this part will be updated] 
cd [...]
```

To use this project, do as the following to install the dependencies.

1. Create a new environment using: `conda create --name myenv python=3.9`.
2. Activate the environment you have just created: `conda activate myenv`.
3. Make the install.sh file executable by running the following command `chmod +x install.sh`.
4. Finally, run the following command to install the required packages inside the conda environment:

```commandline
bash install_conda.sh
```

Or you can create a python environment and install the required packages using `install_pip.sh` file:
1. Create a new environment using: `python -m venv myenv`.
2. Activate the environment you have just created: `source my env/bin/activate`.
3. Make the install.sh file executable by running the following command `chmod +x install.sh`.
4. Finally, run the following command to install the required packages inside the conda environment: 

```commandline
bash install_pip.sh
```

## Prediction

To run the inference code for a pre-trained model on a set of sequences, first you have to set the
`inference_config.yaml` file. You need to have access to the result directory of the pre-trained model
including best checkpoint and config file to be able to run the inference code (refer to pre-trained models section)
The `inference_config.yaml` file is set as the following:

```yaml
pretrained_config_path: /path/to/config.yaml
input_file: /path/to/fasta_file.fasta
output_dir: /path/to/inference/results/
model_dir: /path/to/model/
constrain: True
ensemble: True
with_label: False
```

For data_path, you need to set a fasta file containing the sequences you want to predict on.
Then, run the following command:

```commandline
python predict.py --inference_config_path inference_config.yaml
```

After running the inference code, you can find the results as a json file in the `output_dir` directory specified 
in the `inference_config.yaml` file.

## Pretrained Models

In the following table, you can find the pre-trained models that we have used in the paper. You can download them from
the following links:

| Model Name | Description                 | Download Link                                                                                                                                            |
|------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| MuTarget   | [this part will be updated] | [Download]([this part will be updated]) |
| MuTarget   | [this part will be updated] | [Download]([this part will be updated]) |

We will add more pre-trained models in the future.

## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

[this part will be updated]

```bibtex
@article {Pourmirzaei2024.05.31.596915,
	author = {Pourmirzaei, Mahdi and Esmaili, Farzaneh and Pourmirzaei, Mohammadreza and Wang, Duolin and Xu, Dong},
	title = {Prot2Token: A multi-task framework for protein language processing using autoregressive language modeling},
	year = {2024},
	doi = {10.1101/2024.05.31.596915},
	journal = {bioRxiv}
}
```
