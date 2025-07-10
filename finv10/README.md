# SNA ML Chat Application V7 - The Deep Learning Edition

This is the definitive version of the SNA Chat App, featuring a powerful Deep Learning model (LSTM) for time-series forecasting.

## Key Features

-   **Deep Learning Predictions**: A Long Short-Term Memory (LSTM) neural network is now available for predicting trend lifecycles, offering more sophisticated analysis of time-series data.
-   **Model Showdown**: Compare the performance of three different models directly in the UI: Linear Regression, Random Forest, and LSTM.
-   **Interest-Based Architecture**: The entire platform is built on a foundation of user interests, driving everything from chat room access to the social network graph.
-   **Full SNA Dashboard**: The lobby features an interest-based community graph (Louvain) and ranks users by Degree, Betweenness, and Eigenvector centrality.
-   **Dynamic & Realistic Data**: The database seeder runs a sophisticated simulation to generate chaotic, realistic historical data, providing a rich dataset for the ML/DL models.

## Setup and Installation

1.  **Unzip the File**: Extract this ZIP archive.
2.  **Navigate to the Directory**: `cd SNA_ML_Chat_App_V7_Deep_Learning`
3.  **Create a Virtual Environment**: `python3 -m venv venv` and `source venv/bin/activate`
4.  **Install All Dependencies**: This version requires `tensorflow` for the deep learning model.
    ```bash
    pip install "Flask-SQLAlchemy>=3.0" Flask Flask-Login Flask-SocketIO Flask-Bcrypt pandas scikit-learn eventlet Faker nltk networkx "python-louvain==0.16" tensorflow
    ```
5.  **Seed the Database**: This is a crucial step!
    ```bash
    python3 seed_database.py
    ```
6.  **Run the Application**: `python3 app.py`
7.  **Access the Application**: Open your browser to `http://127.0.0.1:5000`.