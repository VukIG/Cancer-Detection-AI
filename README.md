# Cancer Detection App (CDA) Repository 🚀

## Overview

Welcome to the Cancer Detection App (CDA) repository! This app uses TensorFlow, Matplotlib, NumPy, and Pandas to detect cancer from the Human Against Machine 10000 dataset obtained from Kaggle.

## Dataset

The HAM10000 dataset is organized into the `HAM10000` folder, containing various skin lesion images.

## Data Processing

To handle imbalanced data, we used TensorFlow's `ImageDataGenerator` to generate additional data. The dataset is split into training, validation, and test sets using the `flow_from_directory` function.

## Model Creation

The model is created using TensorFlow's Sequential API, consisting of convolutional layers for image analysis and dense layers for classification.

## Usage

A user-friendly interface is provided through a `while True` loop, allowing users to interactively choose images. The selected image is then loaded, plotted using Matplotlib, and labeled as benign or malignant.

## How to Use

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Cancer-Detection-App.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    python app.py
    ```

## Contributing 🎉

Feel free to contribute, report issues, or just explore the world of cancer detection with joy! 😄

Happy coding! 👩‍💻👨‍💻
