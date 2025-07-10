# 🧪 Reddit-based Social Media SIR Model

This project simulates the spread of information (or "virality") on social media using the **SIR (Susceptible-Infected-Recovered)** model — a popular epidemiological model — applied to **Reddit data**. It also includes a social dashboard and live chat built with **Flask**, showing how topics spread through a community.

---

## 📌 Project Modules

### `finv10/`
A Flask-based **chat room and social media frontend**:
- Real-time chat (`chat.html`)
- User login system
- Dynamic message broadcasting
- Styled with basic CSS and JS

### `redit_01/`
This module handles:
- **Reddit scraping** via `scrape_and_process.py`
- Storage of subreddit data in SQLite (`reddit_data.db`)
- Rendering subreddit insights in `subreddit.html`
- Visualizes how topics might "infect" users over time

### `v10.py`, `v11.py`
Contain the **SIR modeling logic**, simulating how a topic or sentiment spreads among users based on infection (spread), recovery (loss of interest), and susceptibility (potential to engage).

---

## 🧠 Features

- 🔁 Reddit data scraping and analysis
- 🧬 SIR model implementation on real or simulated social data
- 🗣️ Flask-based live chat
- 📊 Web dashboards for subreddit activity
- 🧪 SQLite-based lightweight storage

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/darthvader9092/Reddit_based-Social-media-SIR-Model.git
cd Reddit_based-Social-media-SIR-Model
pip install flask praw matplotlib
cd redit_01
python app.py
cd finv10
python app.py


<pre>
Reddit_based-Social-media-SIR-Model/
├── finv10/               # Flask chat app
│   ├── static/
│   ├── templates/
│   ├── app.py
│   └── database_v7.db
├── redit_01/             # Reddit scraper + dashboard
│   ├── static/
│   ├── templates/
│   ├── scrape_and_process.py
│   ├── app.py
│   └── reddit_data.db
├── v10.py                # SIR model v1
├── v11.py                # SIR model v2
└── README.md
</pre>
