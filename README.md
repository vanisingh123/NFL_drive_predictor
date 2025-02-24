# NFL Prediction Streamlit App
This project implements a machine learning model to predict the points scored per drive in NFL games based on play-by-play data. The model is trained on data from 2000 to 2024 and is deployed using Streamlit to visualize predictions versus actual results for each team.

## Features
- Data Analysis: The app uses historical NFL play-by-play data, including play outcomes, yards gained, drive statistics, and more.
- Predictive Model: A deep learning model is trained to predict the points scored per drive, considering factors like drive length, time of possession, and score differential.
- Model Performance Metrics: The model evaluates accuracy, AUC, precision, recall, and F1 score.
- Interactive Visualization: Streamlit is used to display an interactive plot comparing the predicted and actual points per drive for each team over the years.

## Installation
1. Clone this repository:
```
git clone <repository-url>
cd <repository-folder>
```
2. Install required dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run nfl_prediction.py
```

## Workflow
1. Data Loading & Preprocessing:
- Loads NFL play-by-play data for the years 2000 to 2024 from the nflverse repository.
- Extracts key features such as the number of plays, yards gained, time of possession, and weather conditions.
- Encodes categorical features and scales numerical values.
2. Model Training:
- A deep learning model is trained using Keras (TensorFlow backend) with features like starting yardline, drive time, first downs, and score differential.
- The model is evaluated using metrics like accuracy, AUC, and precision, and the best model is saved.
3. Predictions & Visualization:
- The model is used to predict the average points per drive for each team, and the results are visualized using Plotly in the app.
- Actual vs. predicted points for each season are displayed in an interactive line chart, where users can filter by team.
4. Model Evaluation:
- The app shows the final performance of the model on a test dataset.

## Technologies Used
- Pandas & NumPy: Data manipulation and preprocessing.
- Scikit-learn: Model training, label encoding, and feature scaling.
- TensorFlow/Keras: Deep learning for training and prediction.
- Matplotlib & Plotly: Visualization of model performance and predictions.
- Streamlit: Frontend for displaying interactive charts and metrics.

## Output
Interactive Streamlit app showing:
- Actual vs. predicted average points per drive for each team throughout the seasons
- Output of the predicted number of points scored on a drive given user inputs.

## Contributions
Feel free to fork this repository, make improvements, or suggest new features. Pull requests are welcome!


