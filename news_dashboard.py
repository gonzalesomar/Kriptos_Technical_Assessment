import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('news_preprocessed.csv')
    return df.dropna(subset=['preprocessed_text', 'Label'])

# Use CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Function to get most common words
def get_most_common_words(df, label, n=20):
    counter = Counter(' '.join(df[df['Label'] == label]['preprocessed_text']).split())
    return counter.most_common(n)

# Generate word clouds
def generate_wordcloud(words, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)


#   Title
st.set_page_config(page_title="Kriptos Technical Assessment", page_icon="😎", layout="wide")
local_css("style/style.css")

# Load the data
df = load_data()

# Title
st.title("News Classification Dashboard", anchor='center')

# Sidebar
st.sidebar.header("Parameters")

# Number of most common words to display
n_words = st.sidebar.slider("Number of most common words", 5, 50, 20)

# Main
st.header("Data Overview")
st.write(f"Total Records: {df.shape[0]}")
st.write(f"Total Fake News: {df[df['Label'] == 'Fake'].shape[0]}")
st.write(f"Total Real News: {df[df['Label'] == 'Real'].shape[0]}")

# Pie Chart
st.header("Distribution of News Types")
fig_pie = px.pie(df, names='Label', title='Distribution of Fake and Real News', color_discrete_sequence=['#34D1BF', '#3454D1'])
st.plotly_chart(fig_pie, use_container_width=True)

# Most common words in Fake News
st.header("Most Common Words in Fake News")
most_common_words_fake = get_most_common_words(df, 'Fake', n=n_words)
df_fake_words = pd.DataFrame(most_common_words_fake, columns=['Word', 'Frequency'])
fig_bar_fake = px.bar(df_fake_words, x='Word', y='Frequency', title='Most Common Words in Fake News', color_discrete_sequence=['#6CD4FF'])
st.plotly_chart(fig_bar_fake)


# WordCloud section
st.header("WordCloud")
df_fake = df[df['Label'] == 'Fake']
words_fake = ' '.join(df_fake['preprocessed_text']).split()
counter_fake = Counter(words_fake)
generate_wordcloud(counter_fake, 'Most Frequent Words in Fake News')

# Most common words in Real News
st.header("Most Common Words in Real News")
most_common_words_real = get_most_common_words(df, 'Real', n=n_words)
df_real_words = pd.DataFrame(most_common_words_real, columns=['Word', 'Frequency'])
fig_bar_real = px.bar(df_real_words, x='Word', y='Frequency', title='Most Common Words in Real News', color_discrete_sequence=['#C2EFB3'])
st.plotly_chart(fig_bar_real)

st.header("WordCloud")
df_real = df[df['Label'] == 'Real']
words_real = ' '.join(df_real['preprocessed_text']).split()
counter_real = Counter(words_real)
generate_wordcloud(counter_real, 'Most Frequent Words in Real News')

