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

    print("\n✅ Deep Learning Edition database seeding finished successfully!")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_all()