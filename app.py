import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

# Load data (use your cleaned CSV or original metadata.csv)
@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv')
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    df['abstract'] = df['abstract'].fillna('')
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
    df = df.dropna(subset=['title','publish_time'])
    return df

df = load_data()

# Show sample
st.subheader("Sample of Data")
st.dataframe(df.head())

# Year range slider
years = df['year'].dropna().astype(int)
min_year, max_year = int(years.min()), int(years.max())
year_range = st.slider("Select year range", min_year, max_year, (min_year, max_year))

filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# Plot publications over time
st.subheader("Publications by Year")
year_counts = filtered_df['year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(year_counts.index, year_counts.values)
ax.set_xlabel('Year')
ax.set_ylabel('Count')
st.pyplot(fig)

# Top journals
st.subheader("Top Journals")
top_journals = filtered_df['journal'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax)
ax.set_xlabel('Number of Papers')
ax.set_ylabel('Journal')
st.pyplot(fig)

# Word cloud of titles
st.subheader("Word Cloud of Titles")
titles = ' '.join(filtered_df['title'].dropna().tolist()).lower()
from collections import Counter
import re
words = re.findall(r'\b[a-z]{3,}\b', titles)
common_words = Counter(words).most_common(50)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(common_words))

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)
