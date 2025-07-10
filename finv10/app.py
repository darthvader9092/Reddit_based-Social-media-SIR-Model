import os
import random
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, join_room, leave_room
from flask_bcrypt import Bcrypt
from datetime import datetime
import pandas as pd
import numpy as np

# --- ML / DL Imports ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- SNA / NLP Imports ---
import networkx as nx
import community as community_louvain
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords..."); nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the_deep_learning_finale_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database_v7.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    interests = db.Column(db.String(200), nullable=False)
    def set_password(self, password): self.password_hash = bcrypt.generate_password_hash(password).decode('utf8')
    def check_password(self, password): return bcrypt.check_password_hash(self.password_hash, password)

class Room(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    room_id = db.Column(db.Integer, db.ForeignKey('room.id'))
    user = db.relationship('User')

class TrendSirData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    room_id = db.Column(db.Integer, db.ForeignKey('room.id'))
    topic_id = db.Column(db.Integer)
    s_value = db.Column(db.Integer)
    i_value = db.Column(db.Integer)
    r_value = db.Column(db.Integer)

@login_manager.user_loader
def load_user(user_id): return db.session.get(User, int(user_id))

# --- Analysis Engine ---
class TrendAnalyzer:
    def __init__(self, room_id, n_topics=4):
        self.room_id = room_id
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.messages_df = self._get_messages()

    def _get_messages(self):
        messages = Message.query.filter_by(room_id=self.room_id).all()
        return pd.DataFrame([(m.user_id, m.content) for m in messages], columns=['user_id', 'content']) if messages else pd.DataFrame()

    def analyze_trends(self):
        if len(self.messages_df) < 10: return {"error": "Not enough messages to analyze trends."}
        try:
            tf = self.vectorizer.fit_transform(self.messages_df['content'])
            self.lda.fit(tf)
        except ValueError: return {"error": "Not enough vocabulary to analyze trends."}
        feature_names = self.vectorizer.get_feature_names_out()
        topics = [{"id": i, "keywords": ", ".join([feature_names[w] for w in t.argsort()[:-6:-1]])} for i, t in enumerate(self.lda.components_)]
        doc_topic_dist = self.lda.transform(tf)
        self.messages_df['topic'] = doc_topic_dist.argmax(axis=1)
        first_posts = self.messages_df.drop_duplicates(subset='user_id', keep='first')
        viral_trend_counts = first_posts['topic'].value_counts()
        most_viral_id = viral_trend_counts.idxmax() if not viral_trend_counts.empty else -1
        most_viral_trend = topics[most_viral_id]['keywords'] if most_viral_id != -1 else "None"
        return {"what_is_going_on": topics, "most_infectious_trend": most_viral_trend}

    def _prepare_lstm_data(self, df, look_back=15):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:(i + look_back), :])
            y.append(scaled_data[i + look_back, :])
        return np.array(X), np.array(y), scaler, look_back

    def predict_trend(self, topic_id, days, model_choice='lstm'):
        history = TrendSirData.query.filter_by(room_id=self.room_id, topic_id=topic_id).order_by(TrendSirData.timestamp).all()
        if len(history) < 30: return {"error": f"Not enough historical data for this trend (need 30, have {len(history)})."}

        df = pd.DataFrame([(d.s_value, d.i_value, d.r_value) for d in history], columns=['S', 'I', 'R'])
        feature_names = ['S', 'I', 'R']

        if model_choice == 'lstm':
            X, y, scaler, look_back = self._prepare_lstm_data(df)
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(3)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, batch_size=32, epochs=50, verbose=0)

            last_sequence = scaler.transform(df.tail(look_back))
            current_input = last_sequence.reshape(1, look_back, 3)
            predictions = []
            for _ in range(days):
                pred_scaled = model.predict(current_input, verbose=0)
                pred_unscaled = scaler.inverse_transform(pred_scaled).astype(int)
                predictions.append(np.clip(pred_unscaled[0], 0, None).tolist())
                new_sequence_member = pred_scaled.reshape(1, 1, 3)
                current_input = np.append(current_input[:, 1:, :], new_sequence_member, axis=1)
            return {"predictions": predictions}

        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42) if model_choice == 'random_forest' else LinearRegression()
            X, y = df[feature_names].iloc[:-1], df[feature_names].iloc[1:]
            model.fit(X, y)
            predictions, current_input_df = [], df.iloc[[-1]]
            total_pop = User.query.count()
            for _ in range(days):
                next_pred_array = model.predict(current_input_df[feature_names]).astype(int)
                next_pred_array = np.clip(next_pred_array, 0, total_pop)
                predictions.append(next_pred_array[0].tolist())
                current_input_df = pd.DataFrame(next_pred_array, columns=feature_names)
            return {"predictions": predictions}

def get_interest_graph():
    G = nx.Graph()
    users = User.query.all()
    for u in users: G.add_node(u.username, interests=u.interests)
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1, user2 = users[i], users[j]
            if set(user1.interests.split(',')).intersection(set(user2.interests.split(','))):
                G.add_edge(user1.username, user2.username)
    return G

@app.route('/')
@login_required
def index(): return render_template('index.html', rooms=Room.query.all())

@app.route('/room/<room_name>')
@login_required
def chat_room(room_name):
    room = Room.query.filter_by(name=room_name).first_or_404()
    if room.name not in current_user.interests.split(','):
        flash(f"Access Denied: Your interests do not include '{room.name}'.")
        return redirect(url_for('index'))
    messages = Message.query.filter_by(room_id=room.id).order_by(Message.timestamp.desc()).limit(100).all()
    return render_template('chat.html', room=room, messages=messages[::-1])

@app.route('/api/trends/<room_name>')
@login_required
def get_trends(room_name):
    room = Room.query.filter_by(name=room_name).first()
    if not room: return jsonify({"error": "Room not found"}), 404
    return jsonify(TrendAnalyzer(room.id).analyze_trends())

@app.route('/api/predict_trend/<room_name>', methods=['POST'])
@login_required
def predict_trend_api(room_name):
    data = request.json
    topic_id, days, model_choice = data.get('topic_id'), data.get('days', 7), data.get('model', 'lstm')
    room = Room.query.filter_by(name=room_name).first()
    if not room or topic_id is None: return jsonify({"error": "Invalid request"}), 400
    return jsonify(TrendAnalyzer(room.id).predict_trend(topic_id, days, model_choice))

@app.route('/api/network_graph')
@login_required
def get_network_graph():
    G = get_interest_graph()
    partition = community_louvain.best_partition(G)
    nodes = [{'data': {'id': n, 'community_id': partition.get(n, -1)}} for n in G.nodes()]
    edges = [{'data': {'source': u, 'target': v}} for u, v in G.edges()]
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/api/user_analytics')
@login_required
def get_user_analytics():
    G = get_interest_graph()
    def get_top_5(metric_dict): return sorted(metric_dict.items(), key=lambda item: item[1], reverse=True)[:5]
    degree = get_top_5(nx.degree_centrality(G))
    betweenness = get_top_5(nx.betweenness_centrality(G))
    eigenvector = get_top_5(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-04))
    return jsonify({'degree': degree, 'betweenness': betweenness, 'eigenvector': eigenvector})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        action, username, password = request.form.get('action'), request.form.get('username'), request.form.get('password')
        if action == 'Login':
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password): login_user(user); return redirect(url_for('index'))
            else: flash('Invalid username or password')
        elif action == 'Register':
            if User.query.filter_by(username=username).first(): flash('Username already exists.')
            else:
                interests = "AI,Gaming,Finance,Art".split(',')
                new_user = User(username=username, interests=",".join(random.sample(interests, k=random.randint(1, 2))))
                new_user.set_password(password)
                db.session.add(new_user); db.session.commit()
                flash('Registration successful! Please log in.'); return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout(): logout_user(); return redirect(url_for('login'))

@socketio.on('join')
def on_join(data):
    room_name = data['room']
    if room_name in current_user.interests.split(','):
        join_room(room_name)
        socketio.emit('status', {'msg': f"{current_user.username} has entered the room."}, to=room_name)

@socketio.on('send_message')
def on_send_message(data):
    room_name = data['room']
    room = Room.query.filter_by(name=room_name).first()
    if room and room.name in current_user.interests.split(','):
        msg = Message(content=data['message'], user_id=current_user.id, room_id=room.id)
        db.session.add(msg); db.session.commit()
        socketio.emit('receive_message', {'user': current_user.username, 'msg': data['message']}, to=room_name)

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    socketio.run(app, debug=True)