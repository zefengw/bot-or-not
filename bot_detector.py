import json
import os
import re
import glob
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from textblob import TextBlob

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


FRENCH_STOP_WORDS = [
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "et", "eux", "il", "ils", "je", "la", "le", "les", "leur",
    "lui", "ma", "mais", "me", "mes", "mon", "ne", "nos", "notre", "nous",
    "on", "ont", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa",
    "se", "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu",
    "un", "une", "vos", "votre", "vous", "est", "sont", "suis", "ai",
    "as", "avons", "avez", "fait", "faire", "dit", "cette", "cet",
    "aussi", "bien", "comme", "donc", "encore", "entre", "ici", "jamais",
    "jour", "jours", "lors", "moi", "moins", "ni", "plus", "peut",
    "tout", "tous", "toute", "toutes", "tres", "trop", "rien", "sans",
    "si", "sous", "soi", "ca", "cela", "ceux", "celle", "celles",
    "avoir", "etre", "meme", "autre", "autres", "chaque",
    "apres", "avant", "chez", "contre", "depuis", "alors",
    "deux", "fois", "peu", "ainsi", "car", "dont", "oui", "non",
    "faut", "quand", "sera", "puis", "quoi", "merci",
    "bon", "bonjour", "beaucoup", "comment",
]

LANG_CONFIG = {
    "en": {
        "stop_words": "english",
        "spam_phrases": re.compile(
            r"follow me|check (out|my)|click here|link in bio|subscribe|"
            r"free giveaway|dm me|buy now|limited offer|act now|"
            r"don'?t miss|exclusive deal|sign up",
            re.IGNORECASE,
        ),
    },
    "fr": {
        "stop_words": FRENCH_STOP_WORDS,
        "spam_phrases": re.compile(
            r"suivez[- ]moi|cliquez ici|lien dans (la |ma )?bio|abonne[zr]|"
            r"gratuit|achet[ez]|offre limit|profitez|d[eé]couvr[ei]|"
            r"ne (manquez|ratez) pas|inscri[vt]|cadeau|promo",
            re.IGNORECASE,
        ),
    },
}

URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F]+",
    re.UNICODE,
)


def competition_score(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return 4 * tp - fn - 2 * fp, tp, fn, fp


def find_optimal_threshold(y_true, y_proba):
    best_t, best_s = 0.5, -9999
    for t in np.arange(0.1, 0.95, 0.05):
        s, _, _, _ = competition_score(y_true, (y_proba >= t).astype(int))
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def extract_behavioral_features(posts_df, users_df):
    posts = posts_df.copy()
    posts["created_at"] = pd.to_datetime(posts["created_at"])
    posts = posts.sort_values(["author_id", "created_at"])
    posts["time_diff"] = posts.groupby("author_id")["created_at"].diff().dt.total_seconds()

    stats = posts.groupby("author_id").agg(
        avg_time_diff=("time_diff", "mean"),
        std_time_diff=("time_diff", "std"),
        min_time_diff=("time_diff", "min"),
    ).reset_index().rename(columns={"author_id": "id"})

    df = users_df[["id", "tweet_count", "z_score", "description", "location"]].copy()
    df = df.merge(stats, on="id", how="left").fillna(0)
    df["has_description"] = df["description"].apply(lambda x: 1 if x and len(str(x)) > 0 else 0)
    df["has_location"] = df["location"].apply(lambda x: 1 if x and len(str(x)) > 0 else 0)
    df["desc_len"] = df["description"].apply(lambda x: len(str(x)) if x else 0)

    cols = ["tweet_count", "z_score", "avg_time_diff", "std_time_diff",
            "min_time_diff", "has_description", "has_location", "desc_len"]
    return df[["id"] + cols]


def _tweet_stats(text, spam_re):
    text = str(text)
    alpha = sum(c.isalpha() for c in text)
    upper = sum(c.isupper() for c in text)
    return {
        "n_words": len(text.split()),
        "n_chars": len(text),
        "n_hashtags": len(HASHTAG_RE.findall(text)),
        "n_urls": len(URL_RE.findall(text)),
        "n_mentions": len(MENTION_RE.findall(text)),
        "n_exclamation": text.count("!"),
        "n_question": text.count("?"),
        "n_emoji": len(EMOJI_RE.findall(text)),
        "upper_ratio": upper / alpha if alpha > 0 else 0,
        "is_spam": 1 if spam_re.search(text) else 0,
        "is_rt": 1 if text.strip().startswith("RT @") else 0,
    }


def _user_text_features(group, spam_re):
    texts = group["text"].astype(str).tolist()
    n = len(texts)

    pdf = pd.DataFrame([_tweet_stats(t, spam_re) for t in texts])
    feats = {
        "avg_tweet_length": pdf["n_chars"].mean(),
        "avg_words_per_tweet": pdf["n_words"].mean(),
        "avg_hashtags": pdf["n_hashtags"].mean(),
        "avg_urls": pdf["n_urls"].mean(),
        "avg_mentions": pdf["n_mentions"].mean(),
        "avg_exclamation": pdf["n_exclamation"].mean(),
        "avg_question": pdf["n_question"].mean(),
        "avg_emoji": pdf["n_emoji"].mean(),
        "avg_upper_ratio": pdf["upper_ratio"].mean(),
        "spam_phrase_ratio": pdf["is_spam"].mean(),
        "retweet_ratio": pdf["is_rt"].mean(),
    }

    all_words = " ".join(texts).lower().split()
    feats["lexical_diversity"] = len(set(all_words)) / len(all_words) if all_words else 0

    counts = Counter(texts)
    feats["exact_duplicate_ratio"] = sum(v - 1 for v in counts.values()) / n if n > 1 else 0
    feats["max_repeat_count"] = max(counts.values()) if counts else 0

    if n >= 2:
        word_sets = [set(t.lower().split()) for t in texts]
        pairs = list(combinations(range(len(word_sets)), 2))
        if len(pairs) > 200:
            pairs = [pairs[i] for i in np.random.RandomState(42).choice(len(pairs), 200, replace=False)]
        jaccards = []
        for i, j in pairs:
            inter = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            jaccards.append(inter / union if union else 0)
        feats["avg_jaccard_similarity"] = np.mean(jaccards)
    else:
        feats["avg_jaccard_similarity"] = 0.0

    try:
        polarities = [TextBlob(t).sentiment.polarity for t in texts]
        feats["sentiment_mean"] = np.mean(polarities)
        feats["sentiment_std"] = np.std(polarities) if len(polarities) > 1 else 0.0
    except Exception:
        feats["sentiment_mean"] = 0.0
        feats["sentiment_std"] = 0.0

    return pd.Series(feats)


def extract_text_features(posts_df, lang="en"):
    spam_re = LANG_CONFIG[lang]["spam_phrases"]
    result = posts_df.groupby("author_id").apply(lambda g: _user_text_features(g, spam_re)).reset_index()
    return result.rename(columns={"author_id": "id"})


def extract_tfidf_features(posts_df, lang="en", word_vec=None, char_vec=None, fit=True):
    stop_words = LANG_CONFIG[lang]["stop_words"]
    corpus = posts_df.groupby("author_id")["text"].apply(lambda x: " ".join(map(str, x))).reset_index()

    if fit or word_vec is None:
        word_vec = TfidfVectorizer(max_features=100, stop_words=stop_words, sublinear_tf=True)
        wmat = word_vec.fit_transform(corpus["text"]).toarray()
    else:
        wmat = word_vec.transform(corpus["text"]).toarray()

    if fit or char_vec is None:
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50, sublinear_tf=True)
        cmat = char_vec.fit_transform(corpus["text"]).toarray()
    else:
        cmat = char_vec.transform(corpus["text"]).toarray()

    wcols = [f"tfidf_w{i}" for i in range(wmat.shape[1])]
    ccols = [f"tfidf_c{i}" for i in range(cmat.shape[1])]
    df = pd.DataFrame(np.hstack([wmat, cmat]), columns=wcols + ccols)
    df["id"] = corpus["author_id"].values
    return df, word_vec, char_vec


def build_feature_matrix(posts_df, users_df, lang="en", word_vec=None, char_vec=None, fit=True):
    df_behav = extract_behavioral_features(posts_df, users_df)
    df_text = extract_text_features(posts_df, lang=lang)
    df_tfidf, word_vec, char_vec = extract_tfidf_features(
        posts_df, lang=lang, word_vec=word_vec, char_vec=char_vec, fit=fit
    )

    df = df_behav.merge(df_text, on="id", how="left").merge(df_tfidf, on="id", how="left").fillna(0)
    user_ids = df["id"]
    X = df.drop(columns=["id"])
    return X, user_ids, word_vec, char_vec


def load_dataset(tweet_path, bots_path):
    with open(tweet_path) as f:
        data = json.load(f)
    with open(bots_path) as f:
        bot_ids = set(line.strip() for line in f if line.strip())
    return pd.DataFrame(data["posts"]), pd.DataFrame(data["users"]), bot_ids, data.get("lang")


def load_datasets_by_lang(data_dir, lang):
    files = sorted(glob.glob(os.path.join(data_dir, "dataset.posts&users.*.json")))
    all_posts, all_users, all_bots = [], [], set()

    for pf in files:
        num = os.path.basename(pf).split(".")[-2]
        bf = os.path.join(data_dir, f"dataset.bots.{num}.txt")
        if not os.path.exists(bf):
            continue

        posts, users, bots, ds_lang = load_dataset(pf, bf)
        if ds_lang != lang:
            continue

        posts["author_id"] = posts["author_id"] + f"_ds{num}"
        users["id"] = users["id"] + f"_ds{num}"
        tagged = {b + f"_ds{num}" for b in bots}

        all_posts.append(posts)
        all_users.append(users)
        all_bots.update(tagged)

    if not all_posts:
        raise ValueError(f"No {lang} datasets found in {data_dir}")

    return pd.concat(all_posts, ignore_index=True), pd.concat(all_users, ignore_index=True), all_bots


def train_model(data_dir="files", lang="en", test_size=0.2):
    df_posts, df_users, bot_ids = load_datasets_by_lang(data_dir, lang)
    print(f"[{lang}] {len(df_users)} users, {len(df_posts)} posts, {len(bot_ids)} bots")

    X, user_ids, word_vec, char_vec = build_feature_matrix(df_posts, df_users, lang=lang, fit=True)
    y = user_ids.apply(lambda uid: 1 if str(uid) in bot_ids else 0)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))

    score, tp, fn, fp = competition_score(y_test, y_pred)
    print(f"Competition score @0.5: {score}  (TP={tp} FN={fn} FP={fp})")

    best_t, best_s = find_optimal_threshold(y_test, y_proba)
    print(f"Best threshold: {best_t:.2f} (score={best_s})")

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

    model.fit(X, y)
    return model, word_vec, char_vec, feature_names, best_t


def validate_on_dataset(model, word_vec, char_vec, tweet_path, bots_path, lang, threshold):
    with open(tweet_path) as f:
        data = json.load(f)
    with open(bots_path) as f:
        true_bots = set(line.strip() for line in f if line.strip())

    X, ids, _, _ = build_feature_matrix(
        pd.DataFrame(data["posts"]), pd.DataFrame(data["users"]),
        lang=lang, word_vec=word_vec, char_vec=char_vec, fit=False
    )

    preds = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    y_true = ids.apply(lambda uid: 1 if str(uid) in true_bots else 0)
    score, tp, fn, fp = competition_score(y_true, preds)

    name = os.path.basename(tweet_path)
    print(f"  {name}: score={score}  TP={tp} FN={fn} FP={fp}  (flagged {int(preds.sum())}/{len(ids)})")
    return score, tp, fn, fp


def generate_submission(model, word_vec, char_vec, eval_path, team_name, lang, threshold):
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    with open(eval_path) as f:
        data = json.load(f)

    X, ids, _, _ = build_feature_matrix(
        pd.DataFrame(data["posts"]), pd.DataFrame(data["users"]),
        lang=lang, word_vec=word_vec, char_vec=char_vec, fit=False
    )

    preds = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    flagged = ids[preds == 1]

    filename = f"{team_name}.detections.{lang}.txt"
    with open(filename, "w") as f:
        for uid in flagged:
            f.write(f"{uid}\n")

    print(f"Wrote {filename} ({int(preds.sum())} bots flagged out of {len(ids)} users)")
    return flagged