import os
import zipfile
import textwrap

def create_project_zip():
    """Generates a ZIP file containing the complete Reddit Trend Analyzer project."""
    
    file_contents = {}
    project_name = "redit_01"

    # --- Main Application: app.py ---
    file_contents['app.py'] = """
    import os
    from flask import Flask, render_template, jsonify, request
    from flask_sqlalchemy import SQLAlchemy
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'reddit-trend-analyzer-secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reddit_data.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db = SQLAlchemy(app)

    # --- Database Models ---
    class Subreddit(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), unique=True, nullable=False)

    class Comment(db.Model):
        id = db.Column(db.String(20), primary_key=True)
        subreddit_id = db.Column(db.Integer, db.ForeignKey('subreddit.id'))
        author = db.Column(db.String(100))
        body = db.Column(db.Text)
        created_utc = db.Column(db.DateTime)

    class TrendSirData(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        subreddit_id = db.Column(db.Integer, db.ForeignKey('subreddit.id'))
        topic_id = db.Column(db.Integer)
        date = db.Column(db.Date)
        s_value = db.Column(db.Integer)
        i_value = db.Column(db.Integer)
        r_value = db.Column(db.Integer)

    # --- Analysis Engine ---
    class TrendAnalyzer:
        def __init__(self, subreddit_id, n_topics=5):
            self.subreddit_id = subreddit_id
            self.n_topics = n_topics
            self.vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=10)
            self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            self.comments_df = self._get_comments()

        def _get_comments(self):
            comments = Comment.query.filter_by(subreddit_id=self.subreddit_id).all()
            return pd.DataFrame([(c.body,) for c in comments], columns=['body']) if comments else pd.DataFrame()

        def analyze_trends(self):
            if len(self.comments_df) < 50: return {"error": "Not enough comments to analyze trends."}
            try:
                tf = self.vectorizer.fit_transform(self.comments_df['body'])
                self.lda.fit(tf)
            except ValueError: return {"error": "Not enough vocabulary to analyze trends."}
            
            feature_names = self.vectorizer.get_feature_names_out()
            topics = [{"id": i, "keywords": ", ".join([feature_names[w] for w in t.argsort()[:-6:-1]])} for i, t in enumerate(self.lda.components_)]
            return {"topics": topics}

        def predict_trend(self, topic_id, days):
            history = TrendSirData.query.filter_by(subreddit_id=self.subreddit_id, topic_id=topic_id).order_by(TrendSirData.date).all()
            if len(history) < 15: return {"error": "Not enough historical data for this trend."}
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            df = pd.DataFrame([(d.s_value, d.i_value, d.r_value) for d in history], columns=['S', 'I', 'R'])
            
            X, y = df.iloc[:-1], df.iloc[1:]
            model.fit(X, y)
            
            predictions, current_input_df = [], df.iloc[[-1]]
            total_pop = df.iloc[0].sum() # Total population is the initial S+I+R
            
            for _ in range(days):
                next_pred_array = model.predict(current_input_df).astype(int)
                next_pred_array = np.clip(next_pred_array, 0, total_pop)
                predictions.append(next_pred_array[0].tolist())
                current_input_df = pd.DataFrame(next_pred_array, columns=['S', 'I', 'R'])
            return {"predictions": predictions}

    # --- Flask Routes ---
    @app.route('/')
    def index():
        subreddits = Subreddit.query.all()
        return render_template('index.html', subreddits=subreddits)

    @app.route('/subreddit/<subreddit_name>')
    def subreddit_dashboard(subreddit_name):
        subreddit = Subreddit.query.filter_by(name=subreddit_name).first_or_404()
        return render_template('subreddit.html', subreddit=subreddit)

    @app.route('/api/trends/<subreddit_name>')
    def get_trends(subreddit_name):
        subreddit = Subreddit.query.filter_by(name=subreddit_name).first()
        if not subreddit: return jsonify({"error": "Subreddit not found"}), 404
        return jsonify(TrendAnalyzer(subreddit.id).analyze_trends())

    @app.route('/api/predict_trend/<subreddit_name>', methods=['POST'])
    def predict_trend_api(subreddit_name):
        data = request.json
        topic_id, days = data.get('topic_id'), data.get('days', 7)
        subreddit = Subreddit.query.filter_by(name=subreddit_name).first()
        if not subreddit or topic_id is None: return jsonify({"error": "Invalid request"}), 400
        return jsonify(TrendAnalyzer(subreddit.id).predict_trend(topic_id, days))

    if __name__ == '__main__':
        with app.app_context():
            db.create_all()
        app.run(debug=True)
    """

    # --- Scraper & Processor: scrape_and_process.py ---
    file_contents['scrape_and_process.py'] = """
    import praw
    import pandas as pd
    from datetime import datetime, timedelta
    from app import app, db, Subreddit, Comment, TrendSirData
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    # --- CONFIGURATION ---
    # IMPORTANT: Fill in your Reddit API credentials here
    # Go to: https://www.reddit.com/prefs/apps
    # Click "are you a developer? create an app..."
    # Choose "script" type.
    REDDIT_CLIENT_ID = "YOUR_CLIENT_ID"
    REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
    REDDIT_USER_AGENT = "SNA Trend Scraper by u/YourUsername"

    SUBREDDITS_TO_SCRAPE = ['wallstreetbets', 'technology', 'futurology']
    POST_LIMIT_PER_SUB = 200  # Fetch top 200 posts from the last month
    COMMENTS_PER_POST = 20
    NUM_TOPICS = 5

    def scrape_data():
        if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID":
            print("ERROR: Please fill in your Reddit API credentials in scrape_and_process.py")
            return

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        print("--- Starting Reddit Data Scraping ---")
        for sub_name in SUBREDDITS_TO_SCRAPE:
            print(f"Scraping subreddit: r/{sub_name}")
            subreddit_obj = reddit.subreddit(sub_name)
            
            # Add subreddit to DB if not exists
            sub_db_obj = Subreddit.query.filter_by(name=sub_name).first()
            if not sub_db_obj:
                sub_db_obj = Subreddit(name=sub_name)
                db.session.add(sub_db_obj)
                db.session.commit()

            # Scrape posts and comments
            for post in subreddit_obj.top(time_filter="month", limit=POST_LIMIT_PER_SUB):
                post.comments.replace_more(limit=0) # Expand comment trees
                for comment in post.comments.list()[:COMMENTS_PER_POST]:
                    if comment.author and not Comment.query.get(comment.id):
                        new_comment = Comment(
                            id=comment.id,
                            subreddit_id=sub_db_obj.id,
                            author=str(comment.author),
                            body=comment.body,
                            created_utc=datetime.utcfromtimestamp(comment.created_utc)
                        )
                        db.session.add(new_comment)
            db.session.commit()
            print(f"  - Finished scraping for r/{sub_name}")

    def process_sir_data():
        print("\\n--- Processing Scraped Data to Generate SIR History ---")
        vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=10)
        lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)

        subreddits = Subreddit.query.all()
        for sub in subreddits:
            print(f"Processing SIR data for: r/{sub.name}")
            
            # Clear old SIR data
            TrendSirData.query.filter_by(subreddit_id=sub.id).delete()

            comments = Comment.query.filter_by(subreddit_id=sub.id).all()
            if len(comments) < 50:
                print(f"  - Skipping r/{sub.name}, not enough comments.")
                continue

            df = pd.DataFrame([(c.id, c.author, c.body, c.created_utc.date()) for c in comments], columns=['id', 'author', 'body', 'date'])
            
            # Discover topics
            try:
                tf = vectorizer.fit_transform(df['body'])
                doc_topic_dist = lda.fit_transform(tf)
                df['topic'] = doc_topic_dist.argmax(axis=1)
            except ValueError:
                print(f"  - Skipping r/{sub.name}, not enough vocabulary for topic modeling.")
                continue
            
            all_users = set(df['author'])
            total_users = len(all_users)
            
            # Calculate daily SIR values for each topic
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
            
            for topic_id in range(NUM_TOPICS):
                infected_history = {} # author -> last infected date
                for current_date in date_range:
                    daily_df = df[df['date'] == current_date.date()]
                    
                    # I: Users who commented on this topic today
                    infected_authors_today = set(daily_df[daily_df['topic'] == topic_id]['author'])
                    I = len(infected_authors_today)
                    
                    # Update history for newly infected
                    for author in infected_authors_today:
                        infected_history[author] = current_date
                    
                    # R: Users who were infected previously but not today, and are in a 3-day "cooldown"
                    R = 0
                    for author, last_date in infected_history.items():
                        if author not in infected_authors_today and (current_date - last_date).days <= 3:
                            R += 1
                    
                    # S: Everyone else
                    S = total_users - I - R
                    
                    # Save to DB
                    sir_point = TrendSirData(subreddit_id=sub.id, topic_id=topic_id, date=current_date.date(), s_value=S, i_value=I, r_value=R)
                    db.session.add(sir_point)
            
            db.session.commit()
            print(f"  - Finished processing for r/{sub.name}")

    if __name__ == '__main__':
        with app.app_context():
            db.create_all()
            scrape_data()
            process_sir_data()
        print("\\n✅ Data scraping and processing finished successfully!")
    """

    # --- README.md ---
    file_contents['README.md'] = """
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
    """

    # --- HTML Templates ---
    file_contents['templates/base.html'] = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{% block title %}Reddit Trend Analyzer{% endblock %}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet"><link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}"></head><body><div class="main-wrapper">{% block content %}{% endblock %}</div></body></html>"""
    file_contents['templates/index.html'] = """{% extends "base.html" %}{% block title %}Dashboard{% endblock %}{% block content %}<div class="lobby-container"><header class="lobby-header"><h1>Reddit Trend Analyzer</h1><p>Select a subreddit to analyze its conversation trends.</p></header><div class="room-list-card card"><h2>Analyzed Subreddits</h2><div class="room-list">{% for sub in subreddits %}<a href="{{ url_for('subreddit_dashboard', subreddit_name=sub.name) }}" class="room-item">r/{{ sub.name }}</a>{% else %}<p>No data found. Run the `scrape_and_process.py` script first.</p>{% endfor %}</div></div></div>{% endblock %}"""
    file_contents['templates/subreddit.html'] = """{% extends "base.html" %}{% block title %}r/{{ subreddit.name }}{% endblock %}{% block content %}<div class="analysis-container"><header class="analysis-header"><h1>r/{{ subreddit.name }}</h1><a href="{{ url_for('index') }}" class="back-link">← Back to Dashboard</a></header><div class="dashboard-grid"><div class="trend-analysis-card card"><h3><span class="icon">📈</span>Discovered Trends</h3><button id="analyze-trends-btn" class="btn btn-secondary">Analyze Trends</button><div id="trend-results"><p class="placeholder">Click the button to discover conversation trends.</p></div></div><div class="prediction-card card"><h3><span class="icon">🔮</span>Predict a Trend's Future</h3><div id="prediction-controls"><div class="input-group"><label for="trend-selector">Select Trend</label><select id="trend-selector" disabled><option>Analyze trends first</option></select></div><div class="input-group"><label for="days-input">Prediction Days</label><input type="number" id="days-input" value="7" min="1" max="30" disabled></div></div><button id="predict-btn" class="btn btn-secondary" disabled>Predict</button><div class="chart-container"><canvas id="trend-chart"></canvas></div></div></div></div><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><script src="{{ url_for('static', filename='js/dashboard.js') }}"></script><script>const SUBREDDIT_NAME = {{ subreddit.name|tojson }};</script>{% endblock %}"""
    
    # --- CSS ---
    file_contents['static/css/style.css'] = """:root { --bg-primary: #101727; --bg-secondary: #1E293B; --bg-tertiary: #334155; --text-primary: #F1F5F9; --text-secondary: #94A3B8; --accent-primary: #38BDF8; --accent-secondary: #6366F1; --border-color: #334155; --font-family: 'Poppins', sans-serif; } * { margin: 0; padding: 0; box-sizing: border-box; } body { font-family: var(--font-family); background-color: var(--bg-primary); color: var(--text-primary); } .main-wrapper { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; } .card { background-color: var(--bg-secondary); border-radius: 12px; padding: 2rem; border: 1px solid var(--border-color); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05); } .btn { padding: 0.75rem 1.5rem; border-radius: 8px; border: none; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.2s ease; } .btn-primary { background-color: var(--accent-primary); color: var(--bg-primary); } .btn-primary:hover { background-color: #67cff5; transform: translateY(-2px); } .btn-secondary { background-color: var(--bg-tertiary); color: var(--text-primary); } .btn-secondary:hover { background-color: #475569; } .input-group { margin-bottom: 1.5rem; } .input-group label { display: block; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-secondary); } .input-group input, .input-group select { width: 100%; background-color: var(--bg-primary); border: 1px solid var(--border-color); color: var(--text-primary); padding: 0.75rem; border-radius: 8px; font-size: 1rem; } .input-group input:focus, .input-group select:focus { outline: none; border-color: var(--accent-primary); box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.3); } .lobby-container { width: 100%; max-width: 800px; } .lobby-header { text-align: center; margin-bottom: 2rem; } .room-list { display: flex; flex-direction: column; gap: 0.5rem; } .room-item { background-color: var(--bg-tertiary); padding: 1rem; border-radius: 8px; text-decoration: none; color: var(--text-primary); font-weight: 500; transition: all 0.2s ease; } .room-item:hover { background-color: var(--accent-secondary); transform: translateX(5px); } .analysis-container { width: 100%; max-width: 1400px; } .analysis-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; } .back-link { color: var(--text-secondary); text-decoration: none; } .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; } .trend-analysis-card h3, .prediction-card h3 { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; } #trend-results { margin-top: 1rem; } #trend-results .placeholder { color: var(--text-secondary); } .topic-list { list-style: none; } .topic-list li { background-color: var(--bg-tertiary); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; } #prediction-controls { display: grid; grid-template-columns: 1fr; gap: 1rem; align-items: flex-end; } #prediction-controls + button { width: 100%; margin-top: 1rem; } .chart-container { position: relative; height: 300px; margin-top: 1.5rem; } select:disabled, input:disabled, button:disabled { opacity: 0.5; cursor: not-allowed; }"""
    
    # --- JS ---
    file_contents['static/js/dashboard.js'] = """
    document.addEventListener('DOMContentLoaded', () => {
        const subredditName = SUBREDDIT_NAME; // Injected from template
        const analyzeBtn = document.getElementById('analyze-trends-btn');
        const trendResultsDiv = document.getElementById('trend-results');
        const trendSelector = document.getElementById('trend-selector');
        const daysInput = document.getElementById('days-input');
        const predictBtn = document.getElementById('predict-btn');
        const chartCanvas = document.getElementById('trend-chart');
        let trendChart;

        analyzeBtn.addEventListener('click', async () => {
            trendResultsDiv.innerHTML = '<p class="placeholder">Analyzing... this may take a moment.</p>';
            analyzeBtn.disabled = true;
            const response = await fetch(`/api/trends/${subredditName}`);
            const data = await response.json();
            if (data.error) {
                trendResultsDiv.innerHTML = `<p class="placeholder" style="color: #f87171;">Error: ${data.error}</p>`;
                analyzeBtn.disabled = false;
                return;
            }
            displayTrendResults(data);
            populateTrendSelector(data.topics);
            analyzeBtn.disabled = false;
        });

        function displayTrendResults(data) {
            let html = `<strong>Topics Discovered:</strong><ul class="topic-list">`;
            data.topics.forEach(topic => {
                html += `<li><strong>Topic ${topic.id}:</strong> ${topic.keywords}</li>`;
            });
            html += '</ul>';
            trendResultsDiv.innerHTML = html;
        }

        function populateTrendSelector(topics) {
            trendSelector.innerHTML = '';
            topics.forEach(topic => {
                const option = document.createElement('option');
                option.value = topic.id;
                option.textContent = `Topic ${topic.id}: ${topic.keywords}`;
                trendSelector.appendChild(option);
            });
            trendSelector.disabled = false;
            daysInput.disabled = false;
            predictBtn.disabled = false;
        }

        predictBtn.addEventListener('click', async () => {
            const topicId = parseInt(trendSelector.value);
            const days = parseInt(daysInput.value);
            if (isNaN(topicId) || isNaN(days)) return;
            
            predictBtn.textContent = 'Predicting...';
            predictBtn.disabled = true;
            
            const response = await fetch(`/api/predict_trend/${subredditName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic_id: topicId, days: days })
            });
            const data = await response.json();
            
            if (data.error) {
                alert(`Prediction Error: ${data.error}`);
            } else {
                updateChart(data.predictions);
            }
            
            predictBtn.textContent = 'Predict';
            predictBtn.disabled = false;
        });

        function initializeChart() {
            const ctx = chartCanvas.getContext('2d');
            trendChart = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'Susceptible', data: [], borderColor: '#38BDF8', tension: 0.3, pointBackgroundColor: '#38BDF8' },
                    { label: 'Infected', data: [], borderColor: '#F471B5', tension: 0.3, pointBackgroundColor: '#F471B5' },
                    { label: 'Recovered', data: [], borderColor: '#6B7280', tension: 0.3, pointBackgroundColor: '#6B7280' }
                ]},
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: 'white' }}}, scales: { x: { ticks: { color: 'white' }}, y: { ticks: { color: 'white' }} }}
            });
        }

        function updateChart(predictions) {
            trendChart.data.labels = [];
            trendChart.data.datasets.forEach((dataset) => { dataset.data = []; });
            trendChart.data.labels = Array.from({ length: predictions.length }, (_, i) => `Day ${i + 1}`);
            trendChart.data.datasets[0].data = predictions.map(p => p[0]);
            trendChart.data.datasets[1].data = predictions.map(p => p[1]);
            trendChart.data.datasets[2].data = predictions.map(p => p[2]);
            trendChart.update();
        }

        initializeChart();
    });
    """

    zip_filename = f"{project_name}.zip"
    try:
        print(f"Creating ZIP file: {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as project_zip:
            for file_path, content in file_contents.items():
                archive_path = os.path.join(project_name, file_path)
                project_zip.writestr(archive_path, textwrap.dedent(content).strip())
        print(f"\\n✅ Successfully created {zip_filename}.")
        print("This is the Reddit Trend Analyzer Edition. Please follow the README carefully to set up your Reddit API credentials.")
    except IOError as e:
        print(f"Error creating ZIP file: {e}")

if __name__ == "__main__":
    create_project_zip()
