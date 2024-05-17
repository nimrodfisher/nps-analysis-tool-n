import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk import bigrams

# NLTK resources download
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
custom_stop_words = set(stopwords.words('english'))
words_to_keep = {'lack', 'use', 'often','major','like','poor','easy','well'}  # Words that are meaningful in feedback context
stop_words = custom_stop_words - words_to_keep
# stop_words = set(stopwords.words('english'))


# Initialize NLTK's sentiment analyzer
sia = SentimentIntensityAnalyzer()

def load_data(uploaded_file):
    """Load data from a CSV file and convert date column to datetime."""
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date
    return data

def calculate_nps(df):
    """Assign NPS categories based on scores and add as a new 'Category' column."""
    def nps_category(score):
        if score >= 9:
            return 'Promoter'
        elif score >= 7:
            return 'Passive'
        else:
            return 'Detractor'
    df['Category'] = df['Score'].apply(nps_category)
    return df


def plot_sentiment_analysis(df):
    """Plot the sentiment analysis results with custom colors."""
    if 'Sentiment' in df.columns:
        # Count the occurrences of each sentiment
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Counts']

        # Define custom colors
        color_map = {
            'Positive': 'green',  # Green for Positive
            'Neutral': 'blue',    # Blue for Neutral
            'Negative': 'red'     # Red for Negative
        }

        # Create the bar plot with custom colors
        fig = px.bar(sentiment_counts,
                     x='Sentiment',
                     y='Counts',
                     text='Counts',
                     title='Sentiment Analysis Results',
                     color='Sentiment',
                     color_discrete_map=color_map)  # Applying the custom color map

        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Sentiment data is not available. Please run sentiment analysis first.")
def perform_sentiment_analysis(df):
    """Perform sentiment analysis and add sentiment labels to the dataframe."""
    df['Sentiment'] = df['Feedback'].apply(lambda x: sia.polarity_scores(x)['compound']).apply(
        lambda score: 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
    )
    return df

def plot_nps_trend(df, interval):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    nps_trend = df.groupby([pd.Grouper(freq=interval), 'Category']).size().unstack(fill_value=0)

    # Calculate percentages
    total_responses_per_period = nps_trend.sum(axis=1)
    nps_trend_percentage = nps_trend.div(total_responses_per_period, axis=0) * 100

    # Ensure both Promoter and Detractor columns exist
    if 'Promoter' not in nps_trend_percentage.columns:
        nps_trend_percentage['Promoter'] = 0
    if 'Detractor' not in nps_trend_percentage.columns:
        nps_trend_percentage['Detractor'] = 0

    # Calculate NPS
    nps_trend['NPS'] = nps_trend_percentage['Promoter'] - nps_trend_percentage['Detractor']

    # Plot NPS trend
    fig = px.line(nps_trend, y='NPS', title='NPS Trend Over Time',
                  labels={'index': 'Date', 'value': 'Net Promoter Score'})
    st.plotly_chart(fig, use_container_width=True)

def extract_keywords(df, category):
    """Extract keywords from feedback based on the selected category and calculate their frequency."""
    feedback_text = df[df['Category'] == category]['Feedback'].tolist()
    words = []
    for text in feedback_text:
        word_tokens = word_tokenize(text.lower())
        filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
        words.extend(filtered_words)
    return Counter(words)

def plot_keywords_bubble_chart(keyword_counts, title, num_keywords=20):
    """Display keywords and bi-grams frequency as a bubble chart."""
    if not keyword_counts:
        st.write("No keywords found for this category.")
        return

    keywords_df = pd.DataFrame(keyword_counts.most_common(num_keywords), columns=['Phrase', 'Frequency'])
    fig = px.scatter(keywords_df, x='Phrase', y='Frequency', size='Frequency', color='Phrase', title=title, size_max=num_keywords, hover_data=['Phrase'])
    st.plotly_chart(fig, use_container_width=True)

def extract_keywords_bi_grams(df, category):
    """Extract bi-gram keywords from feedback based on the selected category and calculate their frequency."""
    feedback_text = df[df['Category'] == category]['Feedback'].tolist()
    words = []
    bi_grams_list = []

    for text in feedback_text:
        word_tokens = word_tokenize(text.lower())
        filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]

        # Adding filtered words to the overall list
        words.extend(filtered_words)

        # Create bi-grams from the filtered list of words
        bi_grams = list(bigrams(filtered_words))
        bi_grams_list.extend([' '.join(bi_gram) for bi_gram in bi_grams])

    # Counting single words and bi-grams
    all_keywords = words + bi_grams_list
    return Counter(all_keywords)

def main():
    st.title('NPS Analyzer Tool')
    blog_url = "https://www.linkedin.com/pulse/maximizing-nps-impact-automation-path-enhanced-customer-nimrod-fisher-iiegf/?trackingId=lGvg8yvSQviO%2FtlsC9hHhw%3D%3D"
    st.sidebar.markdown(f"📊 [Read my full blog on NPS analysis!]({blog_url})")
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file (ID, Date, Product, Feedback, Score)", type="csv")

    if uploaded_file:
        data = load_data(uploaded_file)

        if 'Score' in data.columns:
            data = calculate_nps(data)
        else:
            st.error("The uploaded file does not contain a 'Score' column.")
            return

        if 'Category' not in data.columns:
            st.error("Failed to categorize NPS data. Please check the 'calculate_nps' function.")
            return

        interval_options = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
        tabs = st.tabs(["NPS Trend Over Time", "Sentiment Analysis", "Keyword Analysis"])

        with tabs[0]:
            if 'Date' in data.columns and not data['Date'].isna().all():
                selected_interval_label = st.selectbox('Select aggregation interval', list(interval_options.keys()))
                interval = interval_options[selected_interval_label]
                plot_nps_trend(data, interval)
            else:
                st.error("Date column is missing or contains invalid data.")

        with tabs[1]:
            if 'Sentiment' not in data.columns:
                data = perform_sentiment_analysis(data)
            plot_sentiment_analysis(data)

        with tabs[2]:
            category_for_keywords = st.selectbox('Select a category for keywords analysis', ['Promoter', 'Passive', 'Detractor'])
            num_keywords = st.slider("Select number of top phrases", min_value=5, max_value=50, value=20, step=5)
            if st.button('Analyze Keywords'):
                keyword_counts = extract_keywords(data, category_for_keywords)
                if keyword_counts:
                    plot_keywords_bubble_chart(keyword_counts, f'Keyword Frequency for Category {category_for_keywords}', num_keywords)
                else:
                    st.write("No keywords found for this category.")

if __name__ == '__main__':
    main()
