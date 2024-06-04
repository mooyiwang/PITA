# PITA:  Prompting Task Interaction for Argumentation Mining
This repository implements a prompt tuning model for Argumentation Mining (AM), unifying AM's 3 subtasks in a generative way. 

## Requirements
allennlp==2.8.0
allennlp_models==2.10.1
easydict==1.9
nltk==3.8.1
numpy==1.21.6
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
tensorboardX==2.5.1
tensorboardX==2.6.2.2
torch==1.9.0+cu111
torch_geometric==2.1.0.post1
tqdm==4.62.3
transformers==4.26.1
ujson==5.5.0

## Preprocess

### BART-base


### Datasets
The `PE` and `CDCP` datasets have been preprocessed to `.csv` file

Task Interaction Graphs Construction:
(1) w/o task tokens

```sh
python ./data/pe/construct_graphs_2.py
```

(2) w/ task tokens

```sh
python ./data/pe/construct_graphs_6.py
```



## Train & Test

(1) w/o task tokens in prompts, run python scripts with suffix `3` 

```sh
python run_pe3.py --config ./configs/reproduce/pe_bartbase_graph1.json
```

(2) w/ task tokens in prompts, run python scripts with suffix `7_1`

```sh
python run_pe7_1.py --config ./configs/reproduce/pe_bartbase_graph5.json
```


### Reproducibility

We experiment on one Nvidia A100(40G) GPU with CUDA version $11.1$. 
All hyperparameters are in json files.



