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