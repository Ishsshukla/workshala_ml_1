# recommendation_module.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd  # Assuming you are using pandas for DataFrame

def create_similarity_matrix(df):
    df['tags'].fillna(' ', inplace=True)
    text_data = df['tags']

    # Count Vectorizer
    vectorizer = CountVectorizer()
    data_matrix = vectorizer.fit_transform(text_data)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(data_matrix)
    return similarity_matrix

def search_courses(keyword, df):
    matching_courses = df[df['tags'].str.contains(keyword, case=False, na=False)]
    
    if not matching_courses.empty:
        return matching_courses['Title'].tolist()
    else:
        return []
