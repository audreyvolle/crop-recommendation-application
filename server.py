from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('TkAgg')
import multiprocessing

app = Flask(__name__)
Bootstrap(app)

# Load the provided dataset
data = pd.read_csv("Crop_recommendation.csv")

# Train a Naive Bayes model on the entire dataset
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = data[features]
y = data['label']
nb_model = GaussianNB()
nb_model.fit(X, y)

# Function to make predictions
def predict_crop(user_input):
    user_data = pd.DataFrame([user_input], columns=features)
    prediction = nb_model.predict(user_data)
    return prediction[0]

# Function to calculate metrics
def calculate_metrics():
    # Split the data set into training and testing sets
    train_data = data.sample(frac=0.8, random_state=123)
    test_data = data.drop(train_data.index)

    # Train a Naive Bayes model on filtered data
    nb_model_filtered = GaussianNB()
    nb_model_filtered.fit(train_data[features], train_data['label'])

    # Make predictions on the filtered test set
    predictions_filtered = nb_model_filtered.predict(test_data[features])

    roc_curves = []
    auc_values_filtered = []
    for crop in data['label'].unique():
        actual_class = (test_data['label'] == crop).astype(int)
        predicted_class = (predictions_filtered == crop).astype(int)
        fpr, tpr, _ = roc_curve(actual_class, predicted_class)
        roc_curves.append((fpr, tpr))
        auc_values_filtered.append(auc(fpr, tpr))

    return roc_curves, auc_values_filtered

# Function to plot ROC curves
def plot_roc_curves(roc_curves, labels, auc_values_filtered):
    plt.figure(figsize=(8, 8))
    for i in range(len(roc_curves)):
        fpr, tpr = roc_curves[i]
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {auc_values_filtered[i]:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Filtered Data)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ioff()
    return plt

def plot_in_main_thread(roc_curves, labels, auc_values_filtered):
    multiprocessing.Process(target=plot_roc_curves, args=(roc_curves, labels, auc_values_filtered)).start()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'N': float(request.form['N']),
        'P': float(request.form['P']),
        'K': float(request.form['K']),
        'temperature': float(request.form['temperature']),
        'humidity': float(request.form['humidity']),
        'ph': float(request.form['ph']),
        'rainfall': float(request.form['rainfall']),
    }

    prediction = predict_crop(user_input)
    return render_template('index.html', prediction=prediction)

@app.route('/metrics')
def metrics():
    # Call the calculate_metrics function
    roc_curves, auc_values_filtered = calculate_metrics()
    labels = data['label'].unique()

    # Check if auc_values_filtered is defined
    print("auc_values_filtered:", auc_values_filtered)

    plt = plot_roc_curves(roc_curves, labels, auc_values_filtered)

    # Save plot to BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode plot to base64 for displaying in HTML
    plot_base64 = base64.b64encode(img.getvalue()).decode()

    return render_template('metrics.html', auc_values_filtered=auc_values_filtered, labels=labels, plot_base64=plot_base64)

@app.route('/r')
def r_solution():
    return render_template('r.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    plot_in_main_thread(roc_curves, labels, auc_values_filtered)