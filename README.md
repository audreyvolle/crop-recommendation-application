# Crop Recommendation

The Crop Recommendation Application is a tool that helps users determine the most suitable crops to grow based on specific environmental conditions. By providing information such as Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall for a particular area, the application predicts the optimal crops for cultivation.

## How to Run
python server.py

This will start the server, and you can access the application at http://localhost:5001/.

## How to Use
1. Input the environmental conditions of the area where you plan to grow crops.

2. Provide values for Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (in Celsius), Humidity, pH, and Rainfall.

3. Click the "Predict" button to obtain the recommended crops based on the given conditions.
The application will analyze the input data and display the most suitable crops, helping users make informed decisions about crop cultivation.

Feel free to explore the application and discover optimal crop choices for your specific agricultural conditions!

## Metrics
Additionally, you can view metrics related to the model's performance by navigating to http://localhost:5001/metrics. The metrics include values such as Area Under the Curve (AUC), providing insights into the accuracy and effectiveness of the crop recommendation model. Exploring these metrics can offer a deeper understanding of how well the application is performing.

## R Solution
An explanation of how to get the same prediction output in R using Naive Bayes.