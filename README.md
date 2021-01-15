# Disaster Response Pipeline Project

### Summary
In this project a ML model is trained to classify tweets into 36 different categories in the context of quick disaster response. A small Flask webapp is included, where new tweets can be classified using the trained model.

### Files in Repo
data/process_data.py - ETL pipeline
data/disaster/messages.csv - tweets
data/disaster_categories.csv - pre-labeled categories, common id to tweets. 
models/train_classifier.py - ML pipeline
app/run.py - runfile of flask app. template in app/templates. 
ETL Pipeline preparation.ipynb - notebook of ETL pipeline. NOT refactored. refactored code in data/process_data.py
ML Pipeline preparation.ipynb - notebook of ML pipeline. NOT refactored. refactored code in data/train_classifier.py


### Gettings started
This project is written in python 3.7 using sklearn and nltk libraries for the model training. Other packages that need to be installed are numpy, pandas, pickle, sqlalchemy, copy, flask, plotly. 

To train the model and start the app:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
### Contact
For questions, contact deeejy@gmail.com
### Acknowledgements
This project is part of the Udacity Data Scientist Nanodegree. The pre-labeled tweets were provided through Udacity.