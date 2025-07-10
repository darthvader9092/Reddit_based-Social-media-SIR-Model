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
# AFTER (copy and paste this into your file)
# --- CONFIGURATION ---
# IMPORTANT: Fill in your Reddit API credentials here
# Go to: https://www.reddit.com/prefs/apps
# Click "are you a developer? create an app..."
# Choose "script" type.
REDDIT_CLIENT_ID = "WQpy7STdIxK5loYGXuxIkw"
REDDIT_CLIENT_SECRET = "64oDm8jm4-5bIC44cG2bq9sLPCvOkg"
REDDIT_USER_AGENT = "SNA Trend Scraper by u/Few-Comment3494"

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
    print("\n--- Processing Scraped Data to Generate SIR History ---")
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
    print("\n✅ Data scraping and processing finished successfully!")
