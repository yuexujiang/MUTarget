<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

This is the official repository of MuTarget paper.


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

To run the inference code for a pre-trained model on a set of sequences, first you have to have download the pre-trained models and put them under the `result/models` directory (refer to pre-trained models section). Then, run the following command:

```commandline
python predict.py --input_file <your_protein_seq.fa> --output_dir <specify_folder>
```

After running the inference code, you can find the results as a json file in the `output_dir` directory 

## Pretrained Models

In the following table, you can find the pre-trained models that we have used in the paper. You can download them from
the following links and put them under the `results/models` directory

| Model Name | Description                 | Download Link                                                                                                                                            |
|------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| MuTarget   | [ensemble of 5 submodels] | [https://mailmissouri-my.sharepoint.com/:f:/g/personal/yjm85_umsystem_edu/EtxcOvEV07JFrTSA14AWf8oB3TTxNLRsa5-t18iyggIOaw?e=mv5r7t]|


## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

[this part will be updated]

```bibtex
@article {,
	author = {},
	title = {},
	year = {},
	doi = {},
	journal = {}
}
```
