"""
scripts/generate_ato_logs.py
============================
Synthetic dataset generator for the CyberLens Account Takeover (ATO) agent.

What it produces
----------------
data/ato_logs.csv    one row per message:  timestamp, sender, recipient, message
data/ato_labels.csv  one row per account:  account_id, label, scenario
                       label    : 0 = normal, 1 = compromised
                       scenario : which behaviour pattern was simulated
                                  (for error analysis ONLY - never use as a feature)

How the simulation works
------------------------
1. A realistic social graph is generated with networkx (power-law degrees and
   friend clusters, like a real messaging app). Neighbours = "friends".
2. Every account (normal AND compromised) has 30 days of ordinary chatting with
   its friends, with replies coming back. Compromised owners keep chatting too:
   the attacker's activity is ADDED on top, it doesn't replace the owner.
3. The last 7 days are the "observation window". That is when compromised
   accounts get taken over and when some normal accounts do unusual-but-innocent
   things.

Avoiding the data-collection artifact we hit in url_agent.py
-------------------------------------------------------------
- Both classes share the SAME social graph, time range, base activity
  distribution and chat-text pool. Only the takeover behaviour differs.
- Labels are assigned by random shuffle, so account IDs say nothing about labels.
- Deliberate "hard negatives" (normal users who LOOK suspicious) and
  "hard positives" (attackers who look quiet) are included so no single
  feature can separate the classes perfectly:
    normal / broadcaster : sends the same party invite to most friends in minutes
    normal / networker   : messages many strangers over a few days (new job, new city)
    compromised / friend_scam : only messages existing friends (unfamiliar ratio ~0)
    compromised / slow_low    : few, personalised messages spread over days (no burst)

Usage
-----
    python scripts/generate_ato_logs.py                  # defaults: 800 accounts, seed 42
    python scripts/generate_ato_logs.py --accounts 1000 --seed 7
"""

import argparse
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_PATH = DATA_DIR / "ato_logs.csv"
LABELS_PATH = DATA_DIR / "ato_labels.csv"

# ---------------------------------------------------------------------------
# Time layout. OBS_DAYS must match OBS_WINDOW_DAYS in the agent.
# ---------------------------------------------------------------------------
START = datetime(2026, 9, 1)
TOTAL_DAYS = 30
OBS_DAYS = 7
OBS_START = START + timedelta(days=TOTAL_DAYS - OBS_DAYS)
END = START + timedelta(days=TOTAL_DAYS)

# Relative likelihood of sending a message at each hour of the day (people sleep).
HOUR_WEIGHTS = [0.3, 0.15, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 1.8, 1.8, 1.8,
                2.0, 1.8, 1.6, 1.6, 1.7, 1.9, 2.2, 2.5, 2.6, 2.4, 1.8, 0.9]

# ---------------------------------------------------------------------------
# Text pools
# ---------------------------------------------------------------------------
OPENERS = ["hey", "hi", "yo", "hello", "morning", "sup", "heyy", "oi", "hiii"]
TOPICS = [
    "are we still on for lunch", "did you finish the assignment", "check out this song I found",
    "lol that meme was great", "call me when you're free", "what time is practice tomorrow",
    "can you send me the notes", "happy birthday!!", "running 10 min late", "did you watch the match",
    "let's catch up this weekend", "how was the interview", "mom says hi", "where did you park",
    "I'll bring snacks", "the wifi is down again", "can we move the meeting", "thanks for yesterday",
    "good luck for the exam", "send me the photos", "is the canteen open", "who's driving tonight",
    "did you get the parcel", "need help with the lab report", "movie at 9?",
]
CLOSERS = ["", "", "", "!", " :)", " 😂", "?", " tbh", " pls", " bro", " 🙏"]
SHORT_REPLIES = ["ok", "thanks", "lol", "haha", "sure", "on my way", "k", "yes", "no worries", "👍", "??", "wait what"]

INVITES = [
    "hey everyone! housewarming party at my place saturday 8pm, bring snacks",
    "group study session for finals tomorrow in the library at 6, who's in?",
    "we're raising funds for the animal shelter this month, please share!",
    "my band is playing at the cafe this friday night, come through!",
    "reminder: reunion dinner is on sunday, please rsvp so I can book the table",
    "selling my old textbooks before I move out, ping me if you want any",
]
SPAM_TEMPLATES = [
    "Congrats! You've been selected for a $500 gift card, claim it here:",
    "OMG is this you in this video?? look",
    "I just made $3,000 this week with this crypto app, try it:",
    "Your account will be suspended in 24h, verify now:",
    "Free premium subscription for 3 months, grab it before it expires:",
]
FRIEND_SCAMS = [
    "Hey, I'm stuck abroad and lost my wallet, can you send me some money urgently? I'll pay you back tomorrow",
    "hi can you do me a quick favour? I need you to buy a gift card for me, I'll explain later",
    "I'm in a bit of trouble, can you transfer 5000 to me right now? please don't tell anyone",
    "Hey! I accidentally sent a verification code to your number, can you forward it to me please?",
]
SLOW_SCAMS = [
    "I came across your profile and thought you'd like this opportunity",
    "is this you in this photo?",
    "your parcel delivery failed, please reschedule here",
    "we're hiring remote part-time, earn daily from home",
    "you've won a shopping voucher, details here",
]
NETWORKING = [
    "I'm new to the city and joined the running club, would love to connect",
    "saw your post in the alumni group, could we chat about your role sometime?",
    "I'm organising the hackathon next month, would you be interested in mentoring?",
    "found your listing for the apartment, is it still available?",
    "we met at the conference last week, great talk! keeping in touch",
]
NAMES = ["Aarav", "Priya", "Rahul", "Sneha", "Arjun", "Ananya", "Vikram", "Diya", "Rohan", "Meera",
         "Kabir", "Isha", "Aditya", "Neha", "Karan", "Pooja", "Sam", "Alex", "Maria", "John"]
SHORT_LINK_DOMAINS = ["bit.ly", "tinyurl.com", "cutt.ly", "t.co", "rb.gy"]
NORMAL_LINK_DOMAINS = ["youtu.be", "open.spotify.com", "instagram.com/p", "maps.app.goo.gl", "docs.google.com/d"]


class Simulator:
    """Holds the random state and the growing message list."""

    def __init__(self, n_accounts: int, compromised_frac: float, seed: int):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.messages = []  # tuples: (timestamp, sender, recipient, message)

        # --- 1. Social graph -------------------------------------------------
        social = nx.powerlaw_cluster_graph(n_accounts, m=6, p=0.3, seed=seed)
        self.ids = [f"user_{i:04d}" for i in range(n_accounts)]
        self.friends = {self.ids[i]: [self.ids[j] for j in social.neighbors(i)] for i in range(n_accounts)}
        # Each user talks more to some friends than others (close friends vs acquaintances).
        self.friend_weights = {u: list(self.np_rng.exponential(1.0, len(f))) for u, f in self.friends.items()}

        # --- 2. Base activity: ~15% of ALL accounts are low-activity, independent of label
        self.rate = {}
        for u in self.ids:
            if self.rng.random() < 0.15:
                self.rate[u] = float(self.np_rng.lognormal(np.log(0.6), 0.4))
            else:
                self.rate[u] = float(self.np_rng.lognormal(np.log(3.5), 0.5))

        # --- 3. Labels and scenarios, assigned by random shuffle -------------
        shuffled = self.ids[:]
        self.rng.shuffle(shuffled)
        n_comp = int(round(n_accounts * compromised_frac))
        self.labels, self.scenario = {}, {}
        for u in shuffled[:n_comp]:
            self.labels[u] = 1
            self.scenario[u] = self.rng.choices(["blast_spam", "friend_scam", "slow_low"], [0.40, 0.35, 0.25])[0]
        for u in shuffled[n_comp:]:
            self.labels[u] = 0
            self.scenario[u] = self.rng.choices(["regular", "broadcaster", "networker"], [0.80, 0.10, 0.10])[0]

    # ------------------------------------------------------------------ helpers
    def token(self, n=6):
        return "".join(self.rng.choices(string.ascii_letters + string.digits, k=n))

    def name_of(self, account_id):
        return NAMES[int(account_id.split("_")[1]) % len(NAMES)]

    def stranger_for(self, u):
        """Random account that is neither u nor one of u's friends."""
        friends = set(self.friends[u])
        while True:
            r = self.rng.choice(self.ids)
            if r != u and r not in friends:
                return r

    def waking_time(self, day):
        hour = self.rng.choices(range(24), HOUR_WEIGHTS)[0]
        return START + timedelta(days=day, hours=hour, minutes=self.rng.random() * 60)

    def obs_time(self, hours=None):
        """Random moment inside the observation window (optionally restricted to given hours)."""
        day = self.rng.uniform(0, OBS_DAYS - 0.6)
        t = OBS_START + timedelta(days=day)
        if hours is not None:
            t = t.replace(hour=self.rng.choice(hours), minute=self.rng.randint(0, 59))
        return t

    def normal_text(self):
        if self.rng.random() < 0.3:
            return self.rng.choice(SHORT_REPLIES)
        text = f"{self.rng.choice(OPENERS)} {self.rng.choice(TOPICS)}{self.rng.choice(CLOSERS)}"
        if self.rng.random() < 0.08:  # normal people share links too
            text += f" https://{self.rng.choice(NORMAL_LINK_DOMAINS)}/{self.token(8)}"
        return text

    def send(self, ts, sender, recipient, text):
        self.messages.append((ts, sender, recipient, text))

    def maybe_reply(self, original_sender, recipient, ts, p):
        """recipient replies to original_sender with probability p, after a short delay."""
        if self.rng.random() < p:
            delay = timedelta(minutes=float(self.np_rng.exponential(25.0)) + 0.5)
            self.send(ts + delay, recipient, original_sender, self.normal_text())

    # ------------------------------------------------------------ behaviours
    def normal_activity(self, u):
        """Everyday chatting for all 30 days (runs for EVERY account, both classes)."""
        friends, weights = self.friends[u], self.friend_weights[u]
        for day in range(TOTAL_DAYS):
            for _ in range(self.np_rng.poisson(self.rate[u])):
                ts = self.waking_time(day)
                if self.rng.random() < 0.03:  # occasional new contact, a realistic baseline
                    r, p = self.stranger_for(u), 0.30
                else:
                    r, p = self.rng.choices(friends, weights)[0], 0.55
                self.send(ts, u, r, self.normal_text())
                self.maybe_reply(u, r, ts, p)

    def broadcast(self, u, t0):
        """Innocent mass message: same invite to most friends within a few minutes."""
        friends = self.friends[u]
        k = min(len(friends), max(5, int(len(friends) * self.rng.uniform(0.6, 1.0))))
        text = self.rng.choice(INVITES)
        if self.rng.random() < 0.5:
            text += f" https://forms.gle/{self.token(8)}"
        t = t0
        for r in self.rng.sample(friends, k):
            t += timedelta(seconds=self.rng.uniform(3, 20))
            self.send(t, u, r, text)
            self.maybe_reply(u, r, t, 0.65)

    def networker(self, u):
        """Innocent outreach to strangers over several days (new job, new city, selling stuff)."""
        for _ in range(self.rng.randint(8, 25)):
            t = self.obs_time(hours=range(9, 20))
            r = self.stranger_for(u) if self.rng.random() < 0.9 else self.rng.choice(self.friends[u])
            text = f"Hi {self.name_of(r)}, {self.rng.choice(NETWORKING)}"
            if self.rng.random() < 0.3:
                text += f" https://linkedin.com/in/{self.token(8)}"
            self.send(t, u, r, text)
            self.maybe_reply(u, r, t, 0.35)

    def blast_spam(self, u):
        """Attacker blasts one identical link message to friends + many strangers in minutes."""
        text = f"{self.rng.choice(SPAM_TEMPLATES)} http://{self.rng.choice(SHORT_LINK_DOMAINS)}/{self.token(6)}"
        friends = set(self.friends[u])
        targets = list(friends) + [self.stranger_for(u) for _ in range(self.rng.randint(15, 110))]
        targets = list(dict.fromkeys(targets))  # dedupe, keep order
        self.rng.shuffle(targets)
        t = self.obs_time()  # any hour, attackers don't care about our sleep schedule
        for r in targets:
            t += timedelta(seconds=self.rng.uniform(1, 12))
            self.send(t, u, r, text)
            self.maybe_reply(u, r, t, 0.12 if r in friends else 0.02)

    def friend_scam(self, u):
        """Attacker messages ONLY existing friends with an urgent money/OTP request."""
        friends = self.friends[u]
        k = min(len(friends), max(5, int(len(friends) * self.rng.uniform(0.7, 1.0))))
        text = self.rng.choice(FRIEND_SCAMS)
        if self.rng.random() < 0.25:
            text += f" https://{self.rng.choice(SHORT_LINK_DOMAINS)}/{self.token(6)}"
        t = self.obs_time()
        for r in self.rng.sample(friends, k):
            t += timedelta(seconds=self.rng.uniform(5, 60))
            self.send(t, u, r, text)
            self.maybe_reply(u, r, t, 0.40)  # friends often reply "is this really you?"

    def slow_low(self, u):
        """Stealthy attacker: few personalised messages spread over days, mostly to strangers."""
        for _ in range(self.rng.randint(8, 25)):
            t = self.obs_time()
            r = self.stranger_for(u) if self.rng.random() < 0.8 else self.rng.choice(self.friends[u])
            text = f"Hi {self.name_of(r)}, {self.rng.choice(SLOW_SCAMS)}"
            if self.rng.random() < 0.85:
                text += f" http://{self.rng.choice(SHORT_LINK_DOMAINS)}/{self.token(6)}"
            self.send(t, u, r, text)
            self.maybe_reply(u, r, t, 0.06)

    # ------------------------------------------------------------------- run
    def run(self):
        for u in self.ids:
            self.normal_activity(u)
            # ~6% of ALL accounts made an innocent broadcast in the history window,
            # so "has ever mass-messaged" is not a label giveaway.
            if self.rng.random() < 0.06:
                day = self.rng.uniform(0, TOTAL_DAYS - OBS_DAYS - 1)
                self.broadcast(u, START + timedelta(days=day))

            scenario = self.scenario[u]
            if scenario == "broadcaster":
                self.broadcast(u, self.obs_time(hours=range(10, 22)))
            elif scenario == "networker":
                self.networker(u)
            elif scenario == "blast_spam":
                self.blast_spam(u)
            elif scenario == "friend_scam":
                self.friend_scam(u)
            elif scenario == "slow_low":
                self.slow_low(u)

        logs = pd.DataFrame(self.messages, columns=["timestamp", "sender", "recipient", "message"])
        logs = logs[logs["timestamp"] < END]  # replies can spill past the end; drop them
        logs = logs.sort_values("timestamp", kind="stable").reset_index(drop=True)
        logs["timestamp"] = logs["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

        labels = pd.DataFrame({
            "account_id": self.ids,
            "label": [self.labels[u] for u in self.ids],
            "scenario": [self.scenario[u] for u in self.ids],
        })
        return logs, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ATO message logs.")
    parser.add_argument("--accounts", type=int, default=800)
    parser.add_argument("--compromised-frac", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logs, labels = Simulator(args.accounts, args.compromised_frac, args.seed).run()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logs.to_csv(LOGS_PATH, index=False)
    labels.to_csv(LABELS_PATH, index=False)

    print(f"Saved {len(logs):,} messages -> {LOGS_PATH}")
    print(f"Saved {len(labels):,} accounts -> {LABELS_PATH}")
    print("\nAccounts per label / scenario:")
    print(labels.groupby(["label", "scenario"]).size().to_string())


if __name__ == "__main__":
    main()
