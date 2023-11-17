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
from sklearn.manifold import TSNE
import seaborn as sns

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

    correlation_matrix = data[features].corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Plot')
    plt.xlabel('Features')
    plt.ylabel('Features')
    correlation_plot_img = BytesIO()
    plt.savefig(correlation_plot_img, format='png')
    correlation_plot_img.seek(0)
    correlation_plot_base64 = base64.b64encode(correlation_plot_img.getvalue()).decode()

    # Generate t-SNE diagram
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data[features])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=data['label'], palette='viridis')
    plt.title('t-SNE Diagram')
    tsne_plot_img = BytesIO()
    plt.savefig(tsne_plot_img, format='png')
    tsne_plot_img.seek(0)
    tsne_plot_base64 = base64.b64encode(tsne_plot_img.getvalue()).decode()

    # Return all metrics and plots
    return roc_curves, auc_values_filtered, correlation_plot_base64, tsne_plot_base64

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
    roc_curves, auc_values_filtered, correlation_plot_base64, tsne_plot_base64 = calculate_metrics()
    labels = data['label'].unique()
     
    # Extracting the existing plt object from calculate_metrics
    plt = plot_roc_curves(roc_curves, labels, auc_values_filtered)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_base64 = base64.b64encode(img.getvalue()).decode()

    return render_template(
        'metrics.html',
        roc_curves=roc_curves,
        auc_values_filtered=auc_values_filtered,
        labels=labels,
        plot_base64=plot_base64,
        correlation_plot_base64=correlation_plot_base64,
        tsne_plot_base64=tsne_plot_base64
    )

@app.route('/r')
def r_solution():
    return render_template('r.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    plot_in_main_thread(roc_curves, labels, auc_values_filtered)