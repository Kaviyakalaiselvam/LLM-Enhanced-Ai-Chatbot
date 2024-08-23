import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Create a Streamlit app
st.title("Text Summarizer")

# Download NLTK data (sentence tokenizer and stopwords)
nltk.download("punkt")
nltk.download("stopwords")

# Input for the article text
article_text = st.text_area("Enter the text of the article:")

if article_text:
    try:
        # Tokenize the article into sentences
        sentences = sent_tokenize(article_text)

        # Remove stopwords and perform TF-IDF vectorization
        stop_words = "english"  # Use "english" to specify the English stop words
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Calculate sentence similarities using cosine similarity
        sentence_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create a dictionary of sentences and their corresponding sentence scores
        sentence_scores = {}
        for i in range(len(sentences)):
            sentence_scores[i] = sentence_similarity_matrix[i].sum()

        # Sort sentences by score and extract the top sentences as the summary
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:5]  # You can change the number of sentences in the summary

        # Generate the summary text
        summarized_text = " ".join([sentences[i] for i in summary_sentences])

        # Display the summary
        st.write("### Article Summary")
        st.write(summarized_text)

        # Perform sentiment analysis
        st.write("### Sentiment Analysis")
        analysis = TextBlob(article_text)
        sentiment = "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"
        st.write(f"The sentiment of the article is: {sentiment}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
