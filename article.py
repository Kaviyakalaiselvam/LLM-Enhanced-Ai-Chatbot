import streamlit as st
from newspaper import Article
from textblob import TextBlob

# Create a Streamlit app
st.title("Article Summarizer")

# Input for article URL
article_url = st.text_input("Enter the URL of the article:")

if article_url:
    try:
        # Download and parse the article
        article = Article(article_url)
        article.download()
        article.parse()
        article.nlp()

        # Display article information
        st.write("### Article Information")
        st.write(f"**Title:** {article.title}")
        st.write(f"**Authors:** {', '.join(article.authors)}")
        st.write(f"**Publish Date:** {article.publish_date}")

        # Perform text summarization
        st.write("### Article Summary")
        summarized_text = article.summary
        st.write(summarized_text)

        # Perform sentiment analysis
        st.write("### Sentiment Analysis")
        analysis = TextBlob(article.text)
        sentiment = "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"
        st.write(f"The sentiment of the article is: {sentiment}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
