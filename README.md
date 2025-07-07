# xray-classification
Chest X-ray classification using YOLOv8n trained on custom dataset with Roboflow and Google Colab. Classifies Normal, Pneumonia, and COVID-19 cases.

This project uses the YOLOv8n classification model trained on a custom chest X-ray dataset to categorize images into Normal, Pneumonia, and COVID-19 classes. It processes X-ray images to identify key features related to these conditions, enabling efficient and accurate medical image classification. The model can make predictions on individual images and provide classification probabilities for each class.

## xray-classification.py

This script uses a YOLOv8n classification model alongside OpenCV to perform single-image inference on chest X-ray scans.
It loads a chest X-ray image, runs classification using a trained YOLOv8 model, and prints the most probable class label (Normal, Pneumonia, or COVID-19).

The script reads the image with OpenCV, processes it with the YOLOv8 model, extracts prediction probabilities, and returns the predicted class name based on the highest score.

## Dependencies

This project requires the following Python packages. It is recommended to install them in a virtual environment using the `requirements.txt` file included in the repository.
