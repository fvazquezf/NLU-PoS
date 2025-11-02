# Practical Assignment 1 - Part-of-speech (PoS) Tagging

(Natural Language Understanding.
Master in Artificial Intelligence 2025-2026)

Part-of-Speech (POS) tagging is a fundamental step in Natural Language Understanding (NLU), where each word in a sentence is assigned a grammatical category (noun, verb, adjective, etc.).  
This process supports downstream tasks such as parsing, named entity recognition, and machine translation. In this projcet we developed a classifier that given an input text, it give as output the corresponding POS tag for each word. 

## Team Members:

* (Gian Paolo Bulleddu)
* (Yago Estévez Figueiras)
* (Francisco Manuel Vázquez Fernández)

---

## Requirements:
This project runs properly on the image provided by the teaching staff, wich uses Keras 3.

However it is also compatible with previous versions of tensorflow that run with Keras 2.



## Datasets


## Project Structure:
* `posTaggerClass.py`: The main class for creating the model with given parameters, training and evaluate.
* `NLUutils.py`: Utils functions for data preprocessing, creating the models and testing. 
* `notebooks/`: Jupyter notebooks for each language:
    * `EnglishPosTagger.ipynb`
    * `SpanishPosTagger.ipynb`
    * `ItalianPosTagger.ipynb`
* `html folder`: A folder with the results of running each notebook are also provided as example.
* `Datasets`: Datasets for each languages:
    * `UD_English-EWT-master`
    * `UD_Italian-VIT-master`
    * `UD_Spanish-GSD-master`

Souces for the  databases used in this project:

* English: <https://github.com/UniversalDependencies/UD_English-EWT>
* Italian: <https://github.com/UniversalDependencies/UD_Italian-ISDT>
* Spanish: <https://github.com/UniversalDependencies/UD_Spanish-GSD>



## How to Run the code.
* Download the .zip project.
* Run the docker image provided by the teaching staff.
* Launch jupyter lab.
* Open the notebook for the language of interest (spanish, english or italian).
* Execute all cells consecutivelly, they are self explanatory and easy to follow.
  
