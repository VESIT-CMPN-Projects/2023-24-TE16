# Detection of Criminal Activities through CCTV Surveillance

This project implements a real-time violence detection system using computer vision and deep learning techniques. It processes video streams from CCTV cameras or webcams to identify potentially violent situations and sends alerts with location information and captured images.

## Features

- Real-time violence detection using a pre-trained deep learning model
- Face detection in captured images using MTCNN
- Automatic location detection based on IP address
- Telegram bot integration for instant alerts
- Image enhancement for better visibility
- Saving of video footage and captured images

## Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow/Keras
- MTCNN
- Telepot
- Pillow
- Matplotlib
- Requests
- PyTZ

## Installation

1. Clone this repository:
git clone https://github.com/VESIT-CMPN-Projects/2023-24-TE16.git

2. Install required packages

3. Download the pre-trained model file modelnew.h5 and place it in the project directory.

## Configuration

1. Replace 'your bot id' with your Telegram bot token.
2. Replace 'your group id' with your Telegram group or channel ID where you want to receive alerts.

## Usage

Run the main script: location_vio.py

# DATASET FOR TRAINING

https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

We have used only 350 videos from each category (violence and non-violence) out of 1000.

# Video Presentation Link:

https://youtu.be/uBJgyrD01FM
