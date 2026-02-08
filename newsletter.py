# -----------------------------
# Import required libraries
# -----------------------------

import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# Download NLTK resources (run once)
# -----------------------------

nltk.download("stopwords")
nltk.download("vader_lexicon")


# -----------------------------
# Step 1: Load the dataset
# -----------------------------

print("Loading news data...")

news_df = pd.read_csv("data/news.csv")

# Remove missing values and duplicates
news_df.dropna(inplace=True)
news_df.drop_duplicates(inplace=True)


# -----------------------------
# Step 2: Clean the text data
# -----------------------------

print("Cleaning article content...")

stop_words = set(stopwords.words("english"))

def clean_article_text(text):
    """
    Converts text to lowercase,
    removes punctuation and stopwords
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

news_df["clean_content"] = news_df["content"].apply(clean_article_text)


# -----------------------------
# Step 3: Find trending keywords (TF-IDF)
# -----------------------------

print("Extracting trending keywords...")

tfidf = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf.fit_transform(news_df["clean_content"])

trending_keywords = tfidf.get_feature_names_out()


# -----------------------------
# Step 4: Sentiment analysis
# -----------------------------

print("Analyzing sentiment...")

sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Classifies text as Positive, Negative, or Neutral
    """
    score = sentiment_analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

news_df["sentiment"] = news_df["content"].apply(analyze_sentiment)


# -----------------------------
# Step 5: Create short summaries
# -----------------------------

print("Generating summaries...")

def create_summary(text, word_limit=25):
    """
    Creates a short summary from cleaned text
    """
    words = text.split()
    return " ".join(words[:word_limit]) + "..."

news_df["summary"] = news_df["clean_content"].apply(create_summary)


# -----------------------------
# Step 6: Generate HTML newsletter
# -----------------------------

print("Creating HTML newsletter...")

def build_newsletter(dataframe, keywords):
    html_content = """
    <html>
    <head>
        <title>Weekly News Newsletter</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            h1 { color: #2c3e50; }
            h3 { color: #34495e; }
            .article { margin-bottom: 25px; }
            .sentiment { font-style: italic; color: #555; }
            hr { border: 1px solid #ddd; }
        </style>
    </head>
    <body>
    """

    html_content += "<h1>📰 Weekly News Summary</h1>"

    html_content += "<h3>🔥 Trending Topics This Week</h3><ul>"
    for keyword in keywords:
        html_content += f"<li>{keyword}</li>"
    html_content += "</ul><hr>"

    for _, row in dataframe.iterrows():
        html_content += f"""
        <div class="article">
            <h3>{row['title']}</h3>
            <p><b>Source:</b> {row['source']} | <b>Date:</b> {row['date']}</p>
            <p>{row['summary']}</p>
            <p class="sentiment"><b>Sentiment:</b> {row['sentiment']}</p>
        </div>
        <hr>
        """

    html_content += "</body></html>"
    return html_content


newsletter_html = build_newsletter(news_df, trending_keywords)

with open("weekly_newsletter.html", "w", encoding="utf-8") as file:
    file.write(newsletter_html)


# -----------------------------
# Final message
# -----------------------------

print("✅ Newsletter generated successfully!")
print("📄 Open 'weekly_newsletter.html' in your browser")
