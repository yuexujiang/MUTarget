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

1. Create a new environment using: `conda create --name myenv python=3.10`.
2. Activate the environment you have just created: `conda activate myenv`.
3. Make the install.sh file executable by running the following command `chmod +x install.sh`.
4. Finally, run the following command to install the required packages inside the conda environment:

```commandline
sh install.sh
```


## Prediction

To run the inference code for a pre-trained model on a set of sequences, first you have to have download the pre-trained models and put them under the 'result' directory (refer to pre-trained models section). Then, run the following command:

```commandline
python predict.py --input_file <your_protein_seq.fa> --output_dir <specify_folder>
```

After running the inference code, you can find the results as a json file in the `output_dir` directory 

## Pretrained Models

In the following table, you can find the pre-trained models that we have used in the paper. You can download them from
the following links:

| Model Name | Description                 | Download Link                                                                                                                                            |
|------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| MuTarget   | [ensemble of 5 submodels] | [Download]([this part will be updated]) |


## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

[this part will be updated]

```bibtex
@article {Pourmirzaei2024.05.31.596915,
	author = {},
	title = {},
	year = {},
	doi = {},
	journal = {}
}
```
