# Frames24 Repository
This repository contains the dataset and code for the paper __`Mechanistic Interpretability of Socio-Political Frames in Language Models: an Exploration'__ by H. Asghari & S. Nenno presented at the ECML/PKDD workshop on Advances in Interpretable Machine Learning and Artificial Intelligence 2024 (AIMLAI24).

The project investigated two research questions: How well do LLMs _generate_ socio-political frames and _recognize_ their presence in  texts? And can we _localize_ the frames inside LLMs? These questions are explored through four experiments:

- Generation of texts that evoke frames using LLMs and annotating them (on a variety of points) 
- Zeroshot classification of two frames (`strict father`and `nurturing parent`) within rewritten texts
- Localization of SF/NP frames via the [ROME causal tracing](https://rome.baulab.info) method
- Probing for the SF/NP within hidden layer 17 using RFE (detecting 1 key dimension for each) 

The repository is organized according to the above experiments. It contains three datasets (in CSV format).
The first dataset contains 270 generated texts that evoke 10 different socio-political frames 
from three different origins and by five LLMs. 
These texts have been annotated by two annotators (with a third acting as tie-breaker). 
The second dataset is the output of different zeroshot classification techniques on a subset of these texts.  
The third dataset contains SF/NP-v-control data to extract L17 & L21 hidden dims for the last token.
Please see our paper for more explanation about these datasets. 


If you find the dataset, code, or ideas useful, please consider citing our paper.



## 1-Generation-and-Dataset
This folder contains all resources and scripts related to the generation of data and the preparation of datasets for the project.

### Contents
- `story-frame-gen-202403.py`: Script for generating data.
- `data-generated-stories-frames-202505.csv`: Contains generated text data.
- `data-stories-for-probing-202405.csv`: Dataset which contains SF/NP-v-control data to extract L17 & L21 hidden dims for the last token.
- `data-stories-with-labels-202505.csv`: Includes annotations for the generated data.
- `statistics-table-one-202505.ipynb` and `statistics-table-two-202505.ipynb`: Jupyter Notebooks used for generating statistical analyses.

### Usage
To generate data, run the following command:
```bash
python story-frame-gen-202403.py --model llama2 --output gen_stories.csv
```

### Statistical Analysis
Use `statistics-table-one-202505.ipynb` and `statistics-table-two-202505.ipynb` to produce Table 1 and Table 2 for the paper.



## 2-Zeroshot-Classification
Zeroshot classification of two frames (`strict father`and `nurturing parent`) within rewritten texts

### Contents
- `zshot-classification-202505.ipynb`: Notebook to do a zeroshot analysis on our big-dataset of frames and say % each matches SF or NP along with confidence %
- `statistics-table-three-202505.ipynb`: Jupyter Notebook used for generating statistical analyses for table 3

### Statistical Analysis
Use `statistics-table-three-202505.ipynb` to produce Table 3 for the paper.



## 3-Localization-via-ROME
Localization of SF/NP frames via the [ROME causal tracing](https://rome.baulab.info) method

### Contents
- `/rome`: this folder is copied and modified from the original repository , see [rome-repository](https://github.com/kmeng01/rome.git)
- `causal-logistic-202505.ipynb`: Jupyter Notebook used for ROME casual tracing method


## 4-Probing-with-RFE
Probing for the SF/NP within hidden layer 17 using RFE (detecting 1 key dimension for each) 

### Contents 
- `logistic-hidden-SF-202505.ipynb`: Notebook to extract L17 & L21 hidden dims for the last token on Llama3 model with SF/NP-v-control data & run logistic regression models on this data.

## Acknowledgments
We thank Jannes Meyer for organizing the code in this repository.


