# Disaster Response Pipeline Project
### Table of Contents

1. [Introduction](#introduction)
2. [Descriptions](#descriptions)
3. [Getting Started](#getting_started)
4. [Acknowledgements](#acknowledgements)


## Introduction<a name="introduction"></a>
Following a disaster, disaster response organizations typically receive millions of messages either directly or via social media. At those moments, the disaster response organizations normally lack resources to handle the vast amount of information: to filter and select the most information.  

Within a disaster response organization, a different part of taking care of different parts of the problem: one takes care of water, another takes care of blocked roads, and another will be responsible for medical supplies. Therefore an automated workflow that efficiently processes the received messages will be beneficial to direct it to the responsible party.

In this project, we will use prelabeled tweets and text messages to train a machine learning model that is capable of classifying text messages.

## Description<a name="descriptions"></a>
The Project is divided in the following sections:

- ETL Pipeline: extract, clean data and stores the data;
- ML Pipeline: train a model that works to classify text messages;
- Web App: classify the real time input text messages

## Getting Started<a name="getting_started"></a>
### Dependences

### Folder Structure
```
├── LICENSE
├── README.md
├── app
│   ├── run.py  # Flask file that runs the app
│   └── templates
│       ├── go.html # web page contains the classification result
│       └── master.html # main page of web app
├── data
│   ├── DisasterResponse.db # database stores the processed data
│   ├── disaster_categories.csv 
│   ├── disaster_messages.csv
│   └── process_data.py # scripts to clean and store the text messages
├── models
│   └── train_classifier.py # scripts to train the classifier 
└── notebooks
    ├── ETL Pipeline Preparation.ipynb
    └── ML Pipeline Preparation.ipynb
```
### How To:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements <a name="cknowledgements"></a>

- [Figure Eight](https://www.figure-eight.com) provided the prelabeled tweets and text messages used for model training

## Futures
Areas to further optimize in the future:

- Improve visualizations to the web app.
- Customize the design of the web app.
- Deploy the web app to a cloud service provider.
- Improve the efficiency of the code in the ETL and ML pipeline.
- Optimize the model to better handle imbalanced input 
- EDA to further analyze data categories.

## Contact
[Jinchao Chen](tjuchenjinchao@gmail.com)
