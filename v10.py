import os
import zipfile
import textwrap

def create_project_zip():
    """Generates a ZIP file containing the complete, definitive, Deep Learning Flask project."""
    
    file_contents = {}
    project_name = "finv10"

    # --- Main Application: app.py ---
    file_contents['app.py'] = """
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
    """

    # --- Seeder Script: seed_database.py ---
    file_contents['seed_database.py'] = """
    import random
    from faker import Faker
    from datetime import datetime, timedelta
    from app import app, db, User, Room, Message, TrendSirData

    NUM_USERS = 150
    DEFAULT_PASSWORD = "password123"
    INTERESTS = ["AI", "Gaming", "Finance", "Art"]
    NUM_SIM_DAYS = 90

    fake = Faker()
    TOPICS = {
        "AI": [["python", "code", "bug", "feature"], ["ai", "llm", "model", "data"], ["startup", "vc", "funding", "pitch"], ["product", "ux", "design", "user"]],
        "Gaming": [["game", "play", "win", "lose"], ["gpu", "nvidia", "pc", "build"], ["stream", "twitch", "esports", "team"], ["mmo", "rpg", "strategy", "shooter"]],
        "Finance": [["stock", "market", "buy", "sell"], ["options", "calls", "puts", "theta"], ["fed", "rates", "inflation", "economy"], ["tsla", "aapl", "spy", "btc"]],
        "Art": [["art", "gallery", "paint", "canvas"], ["design", "sculpture", "artist", "style"], ["digital", "nft", "procreate", "wacom"], ["history", "museum", "classic", "modern"]]
    }

    def seed_all():
        print("--- Starting Deep Learning Edition Database Seeding (V7) ---")
        db.session.query(Message).delete()
        db.session.query(TrendSirData).delete()
        db.session.query(User).delete()
        db.session.query(Room).delete()
        db.session.commit()

        print("Seeding interest-based rooms...")
        for interest in INTERESTS:
            if not Room.query.filter_by(name=interest).first(): db.session.add(Room(name=interest))
        db.session.commit()

        print("Seeding users with interests...")
        for i in range(NUM_USERS):
            user_interests = random.sample(INTERESTS, k=random.randint(1, 2))
            username = f"{fake.user_name()}_{i}"
            new_user = User(username=username, interests=",".join(user_interests))
            new_user.set_password(DEFAULT_PASSWORD)
            db.session.add(new_user)
        db.session.commit()
        print(f"All users have the default password: '{DEFAULT_PASSWORD}'")

        users = User.query.all()
        rooms = Room.query.all()
        room_map = {room.name: room for room in rooms}
        
        for interest in INTERESTS:
            print(f"--- Simulating Activity for Interest: {interest} ---")
            room = room_map[interest]
            room_topics = TOPICS.get(interest, [])
            num_topics = len(room_topics)
            
            interested_users = [u for u in users if interest in u.interests.split(',')]
            num_interested = len(interested_users)
            user_states = {u.id: {'active_in_topic': None, 'cooldown': 0} for u in interested_users}

            for day in range(NUM_SIM_DAYS):
                daily_posters = random.sample(interested_users, k=random.randint(int(num_interested*0.2), int(num_interested*0.5)))
                for user in daily_posters:
                    topic_id = random.randint(0, num_topics - 1)
                    topic_keywords = room_topics[topic_id]
                    content = " ".join(random.sample(topic_keywords, k=random.randint(2, 4))) + " " + fake.bs()
                    db.session.add(Message(content=content, user_id=user.id, room_id=room.id, timestamp=datetime.utcnow() - timedelta(days=NUM_SIM_DAYS - day)))
                    user_states[user.id]['active_in_topic'] = topic_id
                    user_states[user.id]['cooldown'] = 3

                for topic_id in range(num_topics):
                    I = len([uid for uid, state in user_states.items() if state['active_in_topic'] == topic_id])
                    R = len([uid for uid, state in user_states.items() if state['active_in_topic'] != topic_id and state['cooldown'] > 0])
                    S = num_interested - I - R
                    db.session.add(TrendSirData(room_id=room.id, topic_id=topic_id, s_value=S, i_value=I, r_value=R, timestamp=datetime.utcnow() - timedelta(days=NUM_SIM_DAYS - day)))

                for uid in user_states:
                    user_states[uid]['active_in_topic'] = None
                    if user_states[uid]['cooldown'] > 0: user_states[uid]['cooldown'] -= 1
            
            db.session.commit()
            print(f"  - Finished simulation for {interest}")

        print("\\n✅ Deep Learning Edition database seeding finished successfully!")

    if __name__ == '__main__':
        with app.app_context():
            db.create_all()
            seed_all()
    """

    # --- README.md ---
    file_contents['README.md'] = """
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
    """

    # --- HTML Templates ---
    file_contents['templates/chat.html'] = """
    {% extends "base.html" %}{% block title %}{{ room.name }}{% endblock %}{% block content %}<div class="chat-container"><div class="chat-panel card"><header class="chat-header"><h2>{{ room.name }}</h2><a href="{{ url_for('index') }}" class="back-link">← Back to Lobby</a></header><div id="messages">{% for message in messages %}<div class="message"><span class="username">{{ message.user.username }}:</span><span class="content">{{ message.content }}</span></div>{% endfor %}</div><form id="message-form"><input type="text" id="message-input" placeholder="Type your message here..." autocomplete="off"><button type="submit" class="btn btn-primary">Send</button></form></div><div class="dashboard-panel"><div class="trend-analysis-card card"><h3><span class="icon">📈</span>What's Going On?</h3><button id="analyze-trends-btn" class="btn btn-secondary">Analyze Trends</button><div id="trend-results"><p class="placeholder">Click the button to discover conversation trends.</p></div></div><div class="prediction-card card"><h3><span class="icon">🔮</span>Predict a Trend's Future</h3><div id="prediction-controls"><div class="input-group"><label for="trend-selector">Select Trend</label><select id="trend-selector" disabled><option>Analyze trends first</option></select></div><div class="input-group"><label for="model-selector">Select Model</label><select id="model-selector" disabled><option value="lstm">LSTM (Deep Learning)</option><option value="random_forest">Random Forest (Advanced)</option><option value="linear_regression">Linear Regression (Simple)</option></select></div><div class="input-group"><label for="days-input">Prediction Days</label><input type="number" id="days-input" value="7" min="1" max="30" disabled></div></div><button id="predict-btn" class="btn btn-secondary" disabled>Predict</button><div class="chart-container"><canvas id="trend-chart"></canvas></div></div></div></div><script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><script src="{{ url_for('static', filename='js/chat.js') }}"></script><script>const ROOM_NAME = {{ room.name|tojson }};</script>{% endblock %}
    """
    
    # --- Other files (unchanged but included) ---
    file_contents['templates/index.html'] = """{% extends "base.html" %}{% block title %}Lobby{% endblock %}{% block content %}<div class="lobby-container"><header class="lobby-header"><h1>Community Hub</h1><div class="user-info"><p>Welcome, {{ current_user.username }}!</p><p class="interests-display">Your Interests: {{ current_user.interests }}</p></div><a href="{{ url_for('logout') }}" class="logout-link">Logout</a></header>{% with messages = get_flashed_messages() %}{% if messages %}<div class="flash-messages">{% for message in messages %}<p>{{ message }}</p>{% endfor %}</div>{% endif %}{% endwith %}<div class="lobby-grid"><div class="room-list-card card"><h2>Interest Rooms</h2><p class="card-subtitle">You can only enter rooms matching your interests.</p><div class="room-list">{% for room in rooms %}<a href="{{ url_for('chat_room', room_name=room.name) }}" class="room-item">{{ room.name }}</a>{% else %}<p>No rooms available.</p>{% endfor %}</div></div><div class="centrality-container card"><h2><span class="icon">👑</span> User Analytics</h2><p class="card-subtitle">Identify key players in the network.</p><div class="centrality-controls"><button class="btn btn-tertiary" data-metric="degree">Most Connected</button><button class="btn btn-tertiary" data-metric="betweenness">Key Bridges</button><button class="btn btn-tertiary" data-metric="eigenvector">Influencers</button></div><div id="centrality-results"><p class="placeholder">Select a metric to see top users.</p></div></div><div class="graph-container card"><h2><span class="icon">🕸️</span> Community Network Graph</h2><p class="card-subtitle">Visualize communities based on shared interests.</p><button id="analyze-btn" class="btn btn-secondary">Analyze Communities</button><div id="cy-container"><div id="cy"></div><div id="cy-loading" class="cy-overlay">Click "Analyze" to build the graph.</div></div></div></div></div><script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script><script src="{{ url_for('static', filename='js/lobby.js') }}"></script>{% endblock %}"""
    file_contents['static/css/style.css'] = """:root { --bg-primary: #101727; --bg-secondary: #1E293B; --bg-tertiary: #334155; --text-primary: #F1F5F9; --text-secondary: #94A3B8; --accent-primary: #38BDF8; --accent-secondary: #6366F1; --border-color: #334155; --font-family: 'Poppins', sans-serif; } * { margin: 0; padding: 0; box-sizing: border-box; } body { font-family: var(--font-family); background-color: var(--bg-primary); color: var(--text-primary); } .main-wrapper { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; } .card { background-color: var(--bg-secondary); border-radius: 12px; padding: 2rem; border: 1px solid var(--border-color); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05); } .card-subtitle { color: var(--text-secondary); font-size: 0.9rem; margin-top: -0.75rem; margin-bottom: 1rem; } .btn { padding: 0.75rem 1.5rem; border-radius: 8px; border: none; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.2s ease; } .btn-primary { background-color: var(--accent-primary); color: var(--bg-primary); } .btn-primary:hover { background-color: #67cff5; transform: translateY(-2px); } .btn-secondary { background-color: var(--bg-tertiary); color: var(--text-primary); } .btn-secondary:hover { background-color: #475569; } .btn-tertiary { background-color: transparent; border: 1px solid var(--bg-tertiary); color: var(--text-secondary); } .btn-tertiary:hover { background-color: var(--bg-tertiary); color: var(--text-primary); } .input-group { margin-bottom: 1.5rem; } .input-group label { display: block; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-secondary); } .input-group input, .input-group select { width: 100%; background-color: var(--bg-primary); border: 1px solid var(--border-color); color: var(--text-primary); padding: 0.75rem; border-radius: 8px; font-size: 1rem; } .input-group input:focus, .input-group select:focus { outline: none; border-color: var(--accent-primary); box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.3); } .login-container { width: 100%; max-width: 450px; } .login-box { text-align: center; } .login-box h1 { font-size: 2.5rem; color: var(--accent-primary); margin-bottom: 0.5rem; } .login-box p { color: var(--text-secondary); margin-bottom: 2rem; } .form-actions { display: flex; gap: 1rem; } .form-actions .btn { flex: 1; } .flash-messages { background-color: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; } .lobby-container { width: 100%; max-width: 1200px; } .lobby-header { text-align: center; margin-bottom: 2rem; position: relative; } .user-info .interests-display { color: var(--text-secondary); font-size: 0.9rem; } .logout-link { position: absolute; top: 0; right: 0; color: var(--text-secondary); text-decoration: none; } .logout-link:hover { color: var(--accent-primary); } .lobby-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; gap: 2rem; } .room-list-card { grid-column: 1 / 2; } .centrality-container { grid-column: 2 / 3; } .graph-container { grid-column: 1 / 3; } .room-list { display: flex; flex-direction: column; gap: 0.5rem; } .room-item { background-color: var(--bg-tertiary); padding: 1rem; border-radius: 8px; text-decoration: none; color: var(--text-primary); font-weight: 500; transition: all 0.2s ease; } .room-item:hover { background-color: var(--accent-secondary); transform: translateX(5px); } .chat-container { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; width: 100%; max-width: 1600px; height: 90vh; } .chat-panel { display: flex; flex-direction: column; height: 100%; } .chat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color); } .back-link { color: var(--text-secondary); text-decoration: none; } #messages { flex-grow: 1; overflow-y: auto; padding-right: 1rem; } .message { margin-bottom: 1rem; } .message .username { font-weight: 600; color: var(--accent-primary); margin-right: 0.5rem; } #message-form { display: flex; gap: 1rem; margin-top: 1rem; } #message-form input { flex-grow: 1; } .dashboard-panel { display: flex; flex-direction: column; gap: 2rem; height: 100%; } .dashboard-panel .card { flex-shrink: 0; } .trend-analysis-card h3, .prediction-card h3, .graph-container h2, .centrality-container h2 { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; } #trend-results { margin-top: 1rem; } #trend-results .placeholder, .centrality-container .placeholder { color: var(--text-secondary); } .viral-trend { background-color: rgba(99, 102, 241, 0.2); border: 1px solid var(--accent-secondary); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; } .topic-list { list-style: none; } .topic-list li { background-color: var(--bg-tertiary); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; } #prediction-controls { display: grid; grid-template-columns: 1fr; gap: 1rem; align-items: flex-end; } #prediction-controls + button { width: 100%; margin-top: 1rem; } .chart-container { position: relative; height: 250px; margin-top: 1.5rem; } select:disabled, input:disabled, button:disabled { opacity: 0.5; cursor: not-allowed; } #cy-container { position: relative; height: 400px; border: 1px solid var(--border-color); border-radius: 8px; margin-top: 1rem; background-color: var(--bg-primary); } #cy { width: 100%; height: 100%; } .cy-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; background: rgba(16, 23, 39, 0.8); font-size: 1.5em; color: var(--text-secondary); } .centrality-controls { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; } #centrality-results table { width: 100%; border-collapse: collapse; } #centrality-results th, #centrality-results td { padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color); } #centrality-results th { color: var(--text-secondary); }"""
    file_contents['static/js/lobby.js'] = """document.addEventListener('DOMContentLoaded', () => { const analyzeBtn = document.getElementById('analyze-btn'); const cyContainer = document.getElementById('cy'); const loadingOverlay = document.getElementById('cy-loading'); const centralityControls = document.querySelector('.centrality-controls'); const centralityResultsDiv = document.getElementById('centrality-results'); let cy; let analyticsData = null; const communityColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']; analyzeBtn.addEventListener('click', async () => { loadingOverlay.textContent = 'Building graph...'; loadingOverlay.style.display = 'flex'; analyzeBtn.disabled = true; try { const response = await fetch('/api/network_graph'); const graphData = await response.json(); if (cy) { cy.destroy(); } cy = cytoscape({ container: cyContainer, elements: graphData, style: [ { selector: 'node', style: { 'background-color': (ele) => communityColors[ele.data('community_id') % communityColors.length], 'label': 'data(id)', 'font-size': '10px', 'color': '#fff', 'text-outline-color': '#1E293B', 'text-outline-width': 3 }}, { selector: 'edge', style: { 'width': 1.5, 'line-color': '#334155', 'curve-style': 'bezier' }} ], layout: { name: 'cose', idealEdgeLength: 100, nodeOverlap: 20, refresh: 20, fit: true, padding: 30, randomize: false, componentSpacing: 100, nodeRepulsion: 400000, edgeElasticity: 100, nestingFactor: 5, gravity: 80, numIter: 1000, initialTemp: 200, coolingFactor: 0.95, minTemp: 1.0 } }); } catch (error) { loadingOverlay.textContent = 'Error loading graph.'; } finally { loadingOverlay.style.display = 'none'; analyzeBtn.disabled = false; } }); centralityControls.addEventListener('click', async (e) => { if (e.target.tagName !== 'BUTTON') return; const metric = e.target.dataset.metric; if (!analyticsData) { centralityResultsDiv.innerHTML = '<p class="placeholder">Fetching analytics...</p>'; const response = await fetch('/api/user_analytics'); analyticsData = await response.json(); } displayCentrality(metric); }); function displayCentrality(metric) { const data = analyticsData[metric]; let tableHTML = `<table><thead><tr><th>User</th><th>${metric.charAt(0).toUpperCase() + metric.slice(1)} Score</th></tr></thead><tbody>`; data.forEach(([user, score]) => { tableHTML += `<tr><td>${user}</td><td>${score.toFixed(4)}</td></tr>`; }); tableHTML += '</tbody></table>'; centralityResultsDiv.innerHTML = tableHTML; } });"""
    file_contents['static/js/chat.js'] = """document.addEventListener('DOMContentLoaded', () => { const socket = io(); const roomName = ROOM_NAME; const messagesContainer = document.getElementById('messages'); const messageForm = document.getElementById('message-form'); const messageInput = document.getElementById('message-input'); const analyzeBtn = document.getElementById('analyze-trends-btn'); const trendResultsDiv = document.getElementById('trend-results'); const trendSelector = document.getElementById('trend-selector'); const modelSelector = document.getElementById('model-selector'); const daysInput = document.getElementById('days-input'); const predictBtn = document.getElementById('predict-btn'); const chartCanvas = document.getElementById('trend-chart'); let trendChart; socket.on('connect', () => socket.emit('join', { room: roomName })); socket.on('status', data => addMessage(data.msg, 'status')); socket.on('receive_message', data => addMessage(data.msg, 'user', data.user)); messageForm.addEventListener('submit', e => { e.preventDefault(); const message = messageInput.value.trim(); if (message) { socket.emit('send_message', { room: roomName, message }); messageInput.value = ''; } }); function addMessage(msg, type, user = '') { const div = document.createElement('div'); if (type === 'user') { div.className = 'message'; div.innerHTML = `<span class="username">${user}:</span><span class="content">${msg}</span>`; } else { div.className = 'status-text'; div.textContent = msg; } messagesContainer.appendChild(div); messagesContainer.scrollTop = messagesContainer.scrollHeight; } analyzeBtn.addEventListener('click', async () => { trendResultsDiv.innerHTML = '<p class="placeholder">Analyzing... this may take a moment.</p>'; analyzeBtn.disabled = true; const response = await fetch(`/api/trends/${roomName}`); const data = await response.json(); if (data.error) { trendResultsDiv.innerHTML = `<p class="placeholder" style="color: #f87171;">Error: ${data.error}</p>`; analyzeBtn.disabled = false; return; } displayTrendResults(data); populateTrendSelector(data.what_is_going_on); analyzeBtn.disabled = false; }); function displayTrendResults(data) { let html = `<div class="viral-trend"><strong>Most Infectious Trend:</strong><p>${data.most_infectious_trend}</p></div><strong>Topics Discovered:</strong><ul class="topic-list">`; data.what_is_going_on.forEach(topic => { html += `<li><strong>Topic ${topic.id}:</strong> ${topic.keywords}</li>`; }); html += '</ul>'; trendResultsDiv.innerHTML = html; } function populateTrendSelector(topics) { trendSelector.innerHTML = ''; topics.forEach(topic => { const option = document.createElement('option'); option.value = topic.id; option.textContent = `Topic ${topic.id}: ${topic.keywords}`; trendSelector.appendChild(option); }); trendSelector.disabled = false; modelSelector.disabled = false; daysInput.disabled = false; predictBtn.disabled = false; } predictBtn.addEventListener('click', async () => { const topicId = parseInt(trendSelector.value); const days = parseInt(daysInput.value); const model = modelSelector.value; if (isNaN(topicId) || isNaN(days)) return; predictBtn.textContent = 'Predicting...'; predictBtn.disabled = true; const response = await fetch(`/api/predict_trend/${roomName}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ topic_id: topicId, days: days, model: model }) }); const data = await response.json(); if (data.error) { alert(`Prediction Error: ${data.error}`); } else { updateChart(data.predictions); } predictBtn.textContent = 'Predict'; predictBtn.disabled = false; }); function initializeChart() { const ctx = chartCanvas.getContext('2d'); trendChart = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [ { label: 'Susceptible', data: [], borderColor: '#38BDF8', tension: 0.3, pointBackgroundColor: '#38BDF8' }, { label: 'Infected', data: [], borderColor: '#F471B5', tension: 0.3, pointBackgroundColor: '#F471B5' }, { label: 'Recovered', data: [], borderColor: '#6B7280', tension: 0.3, pointBackgroundColor: '#6B7280' } ]}, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: 'white' }}}, scales: { x: { ticks: { color: 'white' }}, y: { ticks: { color: 'white' }} }} }); } function updateChart(predictions) { trendChart.data.labels = []; trendChart.data.datasets.forEach((dataset) => { dataset.data = []; }); trendChart.data.labels = Array.from({ length: predictions.length }, (_, i) => `Day ${i + 1}`); trendChart.data.datasets[0].data = predictions.map(p => p[0]); trendChart.data.datasets[1].data = predictions.map(p => p[1]); trendChart.data.datasets[2].data = predictions.map(p => p[2]); trendChart.update(); } initializeChart(); });"""
    file_contents['templates/base.html'] = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{% block title %}Trendsetter SNA{% endblock %}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet"><link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}"></head><body><div class="main-wrapper">{% block content %}{% endblock %}</div></body></html>"""
    file_contents['templates/login.html'] = """{% extends "base.html" %}{% block title %}Login{% endblock %}{% block content %}<div class="login-container"><div class="login-box"><h1>Trendsetter SNA</h1><p>Login or Register to analyze community trends.</p>{% with messages = get_flashed_messages() %}{% if messages %}<div class="flash-messages">{% for message in messages %}<p>{{ message }}</p>{% endfor %}</div>{% endif %}{% endwith %}<form method="POST" action="{{ url_for('login') }}"><div class="input-group"><label for="username">Username</label><input type="text" name="username" id="username" required></div><div class="input-group"><label for="password">Password</label><input type="password" name="password" id="password" required></div><div class="form-actions"><button type="submit" name="action" value="Login" class="btn btn-primary">Login</button><button type="submit" name="action" value="Register" class="btn btn-secondary">Register</button></div></form></div></div>{% endblock %}"""

    zip_filename = f"{project_name}.zip"
    try:
        print(f"Creating ZIP file: {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as project_zip:
            for file_path, content in file_contents.items():
                archive_path = os.path.join(project_name, file_path)
                project_zip.writestr(archive_path, textwrap.dedent(content).strip())
        print(f"\\n✅ Successfully created {zip_filename}.")
        print("This is the Grand Finale (FIXED) version. It has been a pleasure building this with you, my friend!")
    except IOError as e:
        print(f"Error creating ZIP file: {e}")

if __name__ == "__main__":
    create_project_zip()
