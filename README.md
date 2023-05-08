# Disaster Response Pipeline Project

### Introduction
Developed disaster response system using NLP and ML techniques to categorize messages and distribute them to relevant organizations for specialized aid. Analyzed pre-labeled disaster data and built a pipeline that returns classification results for new messages. Created Flask app for interactive message classification and disaster data visualization. Further details can be found in [this article](https://medium.com/@runqi/i-built-a-disaster-response-pipeline-and-how-did-i-build-it-7b77e6b91076) on medium.

### File Description
notebooks\
  messages.csv: Raw data of messages sent during disasters, provided by [Figure Eight](https://www.figure-eight.com/)
  categories.csv: Categories of the messages, provided by [Figure Eight](https://www.figure-eight.com/)
  ETL PipelinePreparation.ipynb: Explore and process data 
  ML Pipeline Preparation.ipynb: Build ML pipeline, perform model evaluation and hyperparameter tuning.
  DisasterResponse.db: SQLite Database that stores clean data processed from ETL PipelinePreparation.ipynb
  classifier.pkl: A pickle file that contains the selected model

web_application\
  data\process_data.py: Restructured based on ETL PipelinePreparation.ipynb 
  models\tran_classifier.py: Restructured based on ML Pipeline Preparation.ipynb
  plot_wordcloud.py: Python script to generate wordcloud based on training message
  app\
    static: A folder contain wordcloud_train.png for displaying on the web
    templates: html files used in the web application
    run.py: Python script to launch web application
    
### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Demo

https://user-images.githubusercontent.com/58591088/236385520-f0b8d7f8-2fb6-4ad8-b599-0e5753330e6a.mp4


### Acknowledgements
Udacity for designing the project  
Figure Eight for providing datasets

