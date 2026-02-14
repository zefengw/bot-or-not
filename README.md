# How the bot detector works

The logic lives in `bot_detector.py`. It’s built around a single idea: turn each user into a fixed-length feature vector, then train a Random Forest to predict bot vs human. Everything is language-aware so the same code runs for English and French by switching `lang="en"` or `lang="fr"`.

## Config at the top

`LANG_CONFIG` holds what changes per language: stop words for TF-IDF (sklearn only has built-in English, so French uses a hand-written list) and a regex of spammy phrases (“follow me”, “check out”, “suivez-moi”, “cliquez ici”, etc.). There are also shared regexes for URLs, hashtags, @mentions, and emoji used when computing text stats.

## Scoring and threshold

The competition score is `+4*TP - 1*FN - 2*FP`. `competition_score` takes the true and predicted binary labels and returns that value plus the TP/FN/FP counts. `find_optimal_threshold` loops over probability cutoffs from 0.1 to 0.95 and returns the cutoff that maximizes this score on the given predictions. We use that cutoff later instead of 0.5 so we don’t over-flag humans.

## Feature extraction (three layers)

**Behavioral:** `extract_behavioral_features` expects DataFrames of posts (with `author_id`, `created_at`) and users (with `id`, `tweet_count`, `z_score`, `description`, `location`). It sorts posts by author and time, computes time gaps between consecutive posts per user, then aggregates mean/std/min of those gaps. It merges that with the user table and adds binary flags for having a description/location plus description length. Output is one row per user with those columns plus `id`.

**Text:** For each user we need stats over all their tweets. `_tweet_stats` takes one tweet string and a spam regex and returns things like word count, character count, number of hashtags/URLs/mentions, exclamation/question count, emoji count, fraction of letters that are uppercase, and 0/1 for “contains a spam phrase” and “starts with RT @”. `_user_text_features` gets the list of tweet strings for one user from a groupby, builds a small DataFrame of those per-tweet stats, then takes means for the numeric ones. It also computes lexical diversity (unique words / total words over all tweets), duplicate ratio (how many tweets are exact copies of another), max times any single tweet is repeated, and average pairwise Jaccard similarity between tweets (with up to 200 pairs sampled so it doesn’t blow up). Sentiment is done with TextBlob: polarity per tweet, then mean and std across the user’s tweets. `extract_text_features` just groupbys posts by `author_id` and applies `_user_text_features` with the right language’s spam regex, then renames the index to `id`.

**TF-IDF:** `extract_tfidf_features` first builds one “document” per user by joining all their tweet texts with spaces. It runs two vectorizers on that corpus: a word-level one (100 features, language-specific stop words, sublinear_tf) and a character n-gram one (3–5 grams, 50 features, no stop words). If `fit=True` it fits new vectorizers; if `fit=False` it uses the passed-in fitted ones and only transforms. That way we fit on training data and only transform on validation/eval so the feature space is consistent. The function returns a DataFrame of the combined word + char features plus `id`, and the (possibly updated) word and char vectorizers.

**Putting it together:** `build_feature_matrix` calls the three extractors, merges the three DataFrames on `id`, fills NaNs with 0, and returns the feature matrix (no `id` column), the `id` series, and the two TF-IDF vectorizers. When we’re not training we pass in the vectorizers and `fit=False` so TF-IDF is transform-only.

## Data loading and training

`load_dataset` reads one JSON (posts + users) and one bots txt, returns the two DataFrames, the set of bot IDs, and the `lang` from the JSON. `load_datasets_by_lang` globs for all such JSONs in a directory, for each one finds the matching bots file, loads, and keeps only the ones whose `lang` matches the requested language. So that user IDs don’t collide across datasets it appends `_ds{num}` to every author_id and user id (and to the bot IDs). It concatenates all posts and all users and returns that plus the combined set of tagged bot IDs.

`train_model` uses that to get posts, users, and bot IDs for the given language. It builds the full feature matrix with `fit=True` (so TF-IDF is fitted here), and labels users as 1 if their (tagged) id is in the bot set else 0. It does a stratified 80/20 split, fits a 200-tree Random Forest, runs the classification report and competition score at 0.5, then runs `find_optimal_threshold` on the test-set probabilities and retrains the forest on the full data. The returned threshold is the one that maximized the score. It returns the model, the two fitted vectorizers, the feature names, and that threshold.

## Validation and submission

`validate_on_dataset` is for when you have a dataset with known bot labels. It loads the JSON and bots file, builds the feature matrix with `fit=False` and the vectorizers you pass in, runs the model’s predict_proba, applies the given threshold, and compares to the true labels to compute and print the competition score.

`generate_submission` is for the blind eval set: no bots file. It loads the eval JSON, builds features with the same `fit=False` and the training vectorizers, thresholds the bot probability, and writes the list of flagged user IDs to `{team_name}.detections.{lang}.txt`, one per line.

So the flow is: train once (fit TF-IDF and the forest, get the best threshold), then for any new data always use that same threshold and the same fitted vectorizers so the pipeline is consistent.
