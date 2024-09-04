# Breast-Cancer-Classification-Using-Neural-Networks

A breast cancer classification system using neural networks, designed to identify whether a tumor is benign or malignant. This project avoids using complex deep learning architectures like CNNs, focusing instead on a simple neural network model implemented with TensorFlow and Keras. It is ideal for those who are learning the basics of neural networks and want to understand how traditional fully connected layers work in a medical dataset context.

### Features

- **Data Preprocessing**: Efficiently handles missing values, normalizes feature data using standard scaling, and prepares the dataset for neural network training.
- **Neural Network Model**: Implements a simple multi-layer neural network with one hidden layer and an output layer that classifies the tumor as benign or malignant.
- **Real-time Prediction**: Allows users to input new data points and receive real-time predictions based on the trained model.
- **Basic Evaluation Metrics**: Provides accuracy, confusion matrix, and ROC curve to evaluate the model's performance on unseen test data.

### Tech Stack

- **Programming Languages**: Python
- **Machine Learning Libraries**: TensorFlow, Keras, Scikit-learn
- **Data Processing**:  Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/amandi-udawatta/Breast-Cancer-Classification.git
   cd Breast-Cancer-Classification

2. **Create a Virtual Environment:**:
   ```bash
   python3 -m venv venv

3. **Activate the Virtual Environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
   - On Windows:
     ```bash
     .\venv\Scripts\activate

4. Install the Required Libraries: 
    ```bash
    pip install -r requirements.txt

5. Run the Model:
    ```bash
    python classifier.py

### Authors

- [Amandi Udawatta](https://github.com/amandi-udawatta)
