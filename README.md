# AnimalVision AI 🐾

AnimalVision AI is a custom animal image classifier built using Python, TensorFlow, MobileNetV2, Streamlit, and Hugging Face Spaces.

The model was trained using transfer learning on a Kaggle dataset containing 90 animal classes.

## Live Demo

Try the app here:

https://huggingface.co/spaces/asharansari/animal_vision

## Project Overview

This project was created as a practical implementation of concepts learned during my Generative AI Application Developer course.

The app allows users to upload an animal image and predicts the animal class with a confidence score and top 5 predictions.

## Features

- Upload animal images
- Predict animal class
- Show confidence score
- Show top 5 predictions
- Display animal facts
- Clean Streamlit UI
- Deployed on Hugging Face Spaces

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Transfer Learning
- Streamlit
- Hugging Face Spaces

## Model

The model uses MobileNetV2 as a pre-trained feature extractor. A custom classification head was trained for 90 animal classes.

## Dataset

Dataset used:

Animal Image Dataset - 90 Different Animals  
Kaggle

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
