import pandas as pd
from dotenv import load_dotenv
import numpy as np

import os
from langchain_community.document_loaders import TextLoader  # Takes the raw text (the descriptions) and turn them into a format that the chain can understand
from langchain_text_splitters import CharacterTextSplitter   # Split the document into meaningful chunks
from langchain_openai import OpenAIEmbeddings                # Convert the chunks into document embeddings (using openAI embbdings method)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma                          # Store the embeddings in a database  (Chroma DB)

import gradio as gr

load_dotenv()


# Create a new column to use maximum resolution for covers
books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])


# Vector database
file_path = "tagged_description.txt"
raw_document = TextLoader(file_path, encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_document)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents=documents, embedding=embedding_function)


def retrieve_semantic_recommendation(query: int, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16 ) -> pd.DataFrame: 
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').rstrip(":").split()[0]) for rec in recs]  # Get the ISBN13 of the books
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k) 
    
    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k) 
    else:
        books_recs = books_recs.head(final_top_k)
        
    if tone == "Happy":
        books_recs.sort_values(by="happy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="angry", ascending=False, inplace=True)
    elif tone == "Suspenful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
        
    return books_recs


# Specify what we want to display in the UI
def recommed_book(query : str, category : str, tone: str):
    recommendation = retrieve_semantic_recommendation(query, category, tone) # recommendation db
    results = []
    
    for _,row in recommendation.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."
        
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])}, and {authors_split[-1]}" 
        else:
            authors_str = row["authors"]
        
        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        
        results.append((row["large_thumbnail"], caption))
    return results
        
    
# Creating the UI   
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + [" Happy", "Surprising", "Angry", "Suspenful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book: ",
                                 placeholder= "e.g., A story about love")
        category_dropdown = gr.Dropdown(choices= categories, label = "Select a category", value="All")
        tone_dropdown = gr.Dropdown(choices= tones, label = "Select an emotional tone", value="All")
        submit_button = gr.Button("Get Recommendations")
        
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label= "Recommended books", columns=8, rows=2)
    submit_button.click(fn = recommed_book, inputs= [user_query, category_dropdown, tone_dropdown], outputs= output)
    
if __name__ == "__main__":
    dashboard.launch()
        