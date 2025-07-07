# xray-classification
Chest X-ray classification using YOLOv8n trained on custom dataset with Roboflow and Google Colab. Classifies Normal, Pneumonia, and COVID-19 cases.

This project uses the YOLOv8n classification model trained on a custom chest X-ray dataset to categorize images into Normal, Pneumonia, and COVID-19 classes. It processes X-ray images to identify key features related to these conditions, enabling efficient and accurate medical image classification. The model can make predictions on individual images and provide classification probabilities for each class.

## xray-classification.py

This script uses a YOLOv8n classification model alongside OpenCV to perform single-image inference on chest X-ray scans.
It loads a chest X-ray image, runs classification using a trained YOLOv8 model, and prints the most probable class label (Normal, Pneumonia, or COVID-19).

The script reads the image with OpenCV, processes it with the YOLOv8 model, extracts prediction probabilities, and returns the predicted class name based on the highest score.


## Model and Training Results

The trained model and evaluation outputs (confusion matrix and performance plots) can be accessed from the following Google Drive link:

🔗 Trained Model and Results

Contents of the folder:

`weights/` – Trained YOLOv8n-cls model weights

`results.png` – Training performance metrics over epochs

`confusion_matrix.png` – Raw confusion matrix of the validation set

`confusion_matrix_normalized.png` – Normalized confusion matrix

## Dataset

The model was trained on a custom chest X-ray dataset containing three classes: Normal, Pneumonia, and COVID-19.
You can access the dataset via Roboflow using the link below:

🔗 https://app.roboflow.com/doruk-gunveren/xray-classification-4gdj4/1



## Dependencies

This project requires the following Python packages. It is recommended to install them in a virtual environment using the `requirements.txt` file included in the repository.
