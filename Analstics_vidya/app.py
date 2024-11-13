import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv() 
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

data=pd.read_csv('data.csv')

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# create embidding of text=title+description

data['embedding'] = data['text'].apply(lambda text: embeddings.embed_query(text))

import numpy as np
def recommend_courses(query, df , top=3):
    # Generate embedding for the input query
    query_embedding = embeddings.embed_query(query)

    # cosine similarity 
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])

    # Sort the DataFrame by similarity score 
    recommended_courses = df.sort_values(by='similarity', ascending=False).head(top)
    
    return recommended_courses[['Tile', 'course_description', 'similarity','href','curriculum']]

st.title("🚀 Analytics Vidya Free Course Search Engine 😃")
name =st.text_input("Enter the title/course name you want to search")

if name:
    result=recommend_courses(name,data,3)
    for i in range(len(result)):
        s=result.iloc[i]
        st.header('Top Three Course')
        st.write('👉')
        st.write("Title=",s.iloc[0])
        st.write("Descriptions=,",s.iloc[1])
        st.write("Link=",s.iloc[3])
        l=s.iloc[4]
        st.write("Curriculum")
        for i in l.split("\n\n"):
            for j in i.split("\n"):
                st.write("-",j," ")


