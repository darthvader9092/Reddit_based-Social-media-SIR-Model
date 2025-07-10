# SNA Reddit Trend Analyzer

This application scrapes data from specified Reddit communities (subreddits), uses Natural Language Processing (NLP) to discover conversation trends, and applies the SIR (Susceptible-Infected-Recovered) epidemiological model to analyze and predict the lifecycle of these trends.

## Architecture

1.  **Scraper (`scrape_and_process.py`)**: Connects to the Reddit API to fetch posts and comments. This is run once to populate the database.
2.  **Processor (in `scrape_and_process.py`)**: After scraping, this script analyzes all comments to:
    -   Discover topics (trends) using LDA (Latent Dirichlet Allocation).
    -   Calculate the daily S, I, and R values for each trend based on user comment activity.
    -   Saves this historical SIR data to the database.
3.  **Web App (`app.py`)**: A Flask application that provides a dashboard to:
    -   View the discovered trends for each subreddit.
    -   Use a Machine Learning model (Random Forest) to learn from the historical SIR data and predict the future of a selected trend.

## CRITICAL: Reddit API Setup

This project **will not work** without your own Reddit API credentials.

1.  Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
2.  Scroll to the bottom and click **"are you a developer? create an app..."**.
3.  Fill in the form:
    -   **name**: `MySNAProject` (or anything)
    -   Choose the **"script"** app type.
    -   **description**: (can be blank)
    -   **about url**: (can be blank)
    -   **redirect uri**: `http://localhost:8080`
4.  Click **"create app"**.
5.  You will now see your app listed.
    -   The string of text under the name is your **Client ID**.
    -   The string next to `secret` is your **Client Secret**.
6.  Open the `scrape_and_process.py` file and paste these values into the configuration section at the top. Also, change the `user_agent` to include your Reddit username.

## Setup and Installation

1.  **Unzip the File**: Extract this ZIP archive.
2.  **Navigate to the Directory**: `cd SNA_Reddit_Analyzer`
3.  **Create a Virtual Environment**: `python3 -m venv venv` and `source venv/bin/activate`
4.  **Install All Dependencies**: This version requires `praw` for Reddit scraping.
    ```bash
    pip install "Flask-SQLAlchemy>=3.0" Flask pandas scikit-learn praw nltk
    ```
5.  **Configure and Run the Scraper**:
    -   **Fill in your Reddit API credentials** in `scrape_and_process.py`.
    -   Run the script. This will take several minutes as it fetches and processes a lot of data.
    ```bash
    python3 scrape_and_process.py
    ```
6.  **Run the Web Application**:
    ```bash
    python3 app.py
    ```
7.  **Access the Dashboard**: Open your browser to `http://127.0.0.1:5000`.