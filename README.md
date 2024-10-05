# Peer-to-Peer Federated Learning Framework for Intrusion Detection in Autonomous Vehicles

This repository contains the implementation of a federated learning framework designed to enhance the cybersecurity of autonomous vehicles (AVs). I completed this project as part of my Master's degree in Cybersecurity at Lancaster University.

## Overview

This project leverages a peer-to-peer federated learning approach to create a decentralised intrusion detection system (IDS) to improve detection accuracy and data privacy in AV environments. Unlike traditional centralised IDS approaches, which often suffer from privacy concerns and a single point of failure, the proposed solution utilises a fully decentralised architecture. This architecture allows autonomous vehicles to communicate directly with each other, sharing learned model parameters without transmitting raw data, thereby preserving privacy and resilience.

The core of the IDS is built upon a novel deep learning model, Soft Reordering one-dimensional Convolutional Neural Network (SR-1CNN), designed to efficiently handle Uniform and non-uniform data distribution scenarios.

## Technologies

The following technologies and libraries were used in this project:

Python: The primary language used for the implementation.
PyTorch: A deep learning framework that builds and trains the neural network model.
Torchvision: Provides datasets and tools for image transformation.
NumPy: For numerical computations.
Pandas: For data manipulation and preprocessing.
Scikit-Learn: Used for evaluation metrics and additional machine learning utilities.

## Installation

1. Clone this repository:
   ```bash
       git clone https://github.com/your-username/peer-to-peer-federated-learning-AVs.git

2. Navigate to the project directory:
   cd peer-to-peer-federated-learning-AVs
3. Install the required dependencies:
   pip install -r requirements.txt

The requirements.txt includes the following dependencies:
*   numpy
*   pandas
*   torch
*   torchvision
*   scikit-learn

## Usage

To run the project, use the Python script included in the repository. This script demonstrates training and testing of the proposed federated learning model.

1. Run the script to initiate model training and evaluation:
   ```bash
   python run_NewData_SRCNN.py
2. Update the paths in the script to point to your data files.

**Note: Ensure that you have installed all the required dependencies by running:**
        pip install -r requirements.txt

## Features
--------

*   **Decentralised Intrusion Detection**: A federated learning-based IDS designed specifically for autonomous vehicle environments with peer-to-peer communication.
*   **Deep Learning Model**: Uses a novel Soft Reordering one-dimensional Convolutional Neural Network (SR-1CNN) to enhance detection accuracy.
*   **Privacy-Preserving**: The model parameters are shared rather than raw data, ensuring data privacy.
*   **Robust Against Failures**: Unlike client-server federated learning, this framework is resilient to single points of failure using a fully peer-to-peer approach.

## Datasets
--------

The datasets used in this project for training and evaluation include:

*   **NSL-KDD**: A well-known dataset for intrusion detection research.
*   **Car Hacking Dataset**: A dataset specifically designed to address security issues in autonomous vehicle environments.

**Note**: The datasets are not included in this repository. Please refer to their respective sources for more information on obtaining these datasets.

## Background
----------

This project was part of my Master's thesis at Lancaster University, where I pursued a Master's in Cybersecurity. The main objective of the research was to propose and implement a solution for intrusion detection in autonomous vehicles, using a novel federated learning approach to address privacy and resilience concerns associated with traditional centralised systems.



