# ml-predictions-project

The purpose of this project was to use various machine learning models and historical weather data to predict whether rain will occur on a given day. First, Python with Pandas was used to clean the data and transform it for use with the four selected machine learning models: Logistic Regression, KNN, Random Forests, and SVM. Python, with Pandas and Sklearn, was used to develop the machine learning models and refine the data using feature engineering. Finally, HTML, JavaScript, and Python with Flask were used to build an interactive web page for user queries of the machine learning models.

## Questions

1. What is the maximum score of each machine learning model?
2. What are the rain predictions from each machine learning model?
3. What is the overall rain prediction based on the model scores and individual model predictions?

## Datasets

1. https://github.com/jherberg462/ml-predictions-project/blob/master/aus_weather/weatherAUS.csv
2. https://github.com/jherberg462/ml-predictions-project/blob/master/aus_weather/weatherAUS_clean.csv
3. https://github.com/jherberg462/ml-predictions-project/blob/master/aus_weather/weatherAUS_feature_engineer.csv

## Tasks

### Machine Learning Model Development

1. Import, clean, and transform the raw data.
2. Build the initial machine learning models with default parameters.
3. Refine the machine learning models with grid search.
4. Use feature engineering to further transform the cleaned dataset.
5. Re-build the machine learning models with the feature engineered dataset.

### Flask App & Web Page Development

1. Build a web page with a drop-down menu for user inputs and a button to pass the inputs to a Flask application.
2. Create a Flask application route to accept the user inputs from the web page.
3. Import the feature engineered data and generate a scaling factor for the input data.
4. Import the machine learning models.
5. Scale the input data and pass the scaled data to the machine learning models.
6. Output the machine learning model results to the web page for display.

## Results

1. https://weather-predictions-ml.herokuapp.com/