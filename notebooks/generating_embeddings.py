# %%
import torch
import os
import pyarrow.parquet as pa
import pickle
from tqdm import tqdm


# text stuff
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer

# mathy stuff
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# plotting stuff 
import matplotlib.pyplot as plt
# %%
os.chdir("/teamspace/studios/this_studio/mapping_reform")

# %%
# function to filter and clean publication dfs
# first setting articles' min words externally
min_words = 5
def clean_df(df_raw):
    # keeping only rows classified as 'text', excluding for e.g. titles
    # keeping issue_id and content columns
    df = df_raw[df_raw['class'] == 'text'][['issue_id', 'content']].copy()

    # creating datetime col from issue_id, using regex to extract yyyy-mm-dd and setting NaT if no valid format
    df['pub_date'] = pd.to_datetime(
        df['issue_id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0],
        format='%Y-%m-%d', 
        errors='coerce')

    # keeping articles with >5 words, as they are likely not significant article bodies otherwise
    content_lens = df['content'].str.count(r'\w+')
    short_articles = df[content_lens < min_words]
    df = df[content_lens >= min_words]
    print(f"{len(short_articles)} articles dropped - fewer than {min_words} words.")

    # replacing newline symbols that i noticed in the process with spaces
    df['content'] = df['content'].str.replace('\n', ' ', regex=False)

    return df.reset_index(drop = True), short_articles.reset_index(drop=True)

# %%
# publication parquet files
parquet_files = {
    "The English Woman's Journal": "data/post_processed/English_Womans_Journal_issue_PDF_files.parquet",
    "The Tomahawk": "data/post_processed/Tomahawk_issue_PDF_files.parquet",
    "The Leader": "data/post_processed/Leader_issue_PDF_files.parquet",
    "The Monthly Repository": "data/post_processed/Monthly_Repository_issue_PDF_files.parquet",
    "The Northern Star": "data/post_processed/Northern_Star_issue_PDF_files.parquet",
    "The Publisher's Circular": "data/post_processed/Publishers_Circular_issue_PDF_files.parquet"
}
# %%
# defining model
model = SentenceTransformer("tomaarsen/static-retrieval-mrl-en-v1")

# %% 
# function for embedding in batches to prevent kernel crashing
def batched_encode(texts, model, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# %% northern star
# separating the northern star because of downstream problems with reading in the pickled file, hence trying again to pickle it
star_raw = pa.read_table("data/post_processed/Northern_Star_issue_PDF_files.parquet").to_pandas()
star, star_dropped = clean_df(star_raw)
star_text = star["content"].tolist()

# %%
star_vec = model.encode(star_text, show_progress_bar=True, convert_to_numpy=True)

# %%
star_filepath = os.path.join("data/pub_embeddings", "the_northern_star.pkl")
print(f"Saving embeddings for The Northern Star to {star_filepath}...")
with open(star_filepath, "wb") as f:
    pickle.dump({"articles": star_text, "embeddings": star_vec}, f)

# %% making dfs from parquets 
# dict to store read+filtered dataframes
cleaned_dfs = {}
short_articles_dfs = {}

for key, path in parquet_files.items():
    print(f"Loading and filtering: {key}")

    # reading in parquet file to a pandas df
    raw_df = pa.read_table(path).to_pandas()

    # passing raw df to filtering function defined above
    cleaned_df, short_df = clean_df(raw_df)

    # i found that there's a misformatted date, '1852-31-07', in The Leader's file. fixing this below:
    if key == "The Leader":
        cleaned_df.loc[cleaned_df['pub_date'] == '1852-31-07', 'pub_date'] = '1852-07-31'

    cleaned_dfs[key] = cleaned_df 
    short_articles_dfs[key] = short_df


# %%
# assigning outputs to individual publication variables
ewj = cleaned_dfs["The English Woman's Journal"]
thawk = cleaned_dfs["The Tomahawk"]
leader = cleaned_dfs["The Leader"]
monrepo = cleaned_dfs["The Monthly Repository"]
star = cleaned_dfs["The Northern Star"]
circ = cleaned_dfs["The Publisher's Circular"]

ewj_dropped = short_articles_dfs["The English Woman's Journal"]
thawk_dropped = short_articles_dfs["The Tomahawk"]
leader_dropped = short_articles_dfs["The Leader"]
monrepo_dropped = short_articles_dfs["The Monthly Repository"]
star_dropped = short_articles_dfs["The Northern Star"]
circ_dropped = short_articles_dfs["The Publisher's Circular"]  

# %%
# directory to store embeddings 
embedding_dir = "data/pub_embeddings"

if not os.path.exists(embedding_dir):
    print(f"Creating directory to store embeddings: {embedding_dir}")
    os.makedirs(embedding_dir)
else:
    print(f"Directory for embeddings already exists: {embedding_dir}")

# initialising dictionary for publication embeddings
embeddings = {}

for pub, pub_df in cleaned_dfs.items():
    if pub != "star":
        filename = pub.lower().replace(" ", "_").replace("'", "").replace(".","") + ".pkl"
        filepath = os.path.join(embedding_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Generating vector embeddings for {pub}...")
            pub_text = pub_df["content"].tolist()
            pub_vec = model.encode(pub_text, show_progress_bar=True, convert_to_numpy=True)
            print(f"Vector array of shape {pub_vec.shape} obtained.")
            print(f"Saving embeddings for {pub} to {filepath}...")
            with open(filepath, "wb") as f:
                pickle.dump({"articles": pub_text, "embeddings": pub_vec}, f)

        else:
            print(f"Pre-computed embeddings already exist for {pub}. Loading them from {filepath}...")
            with open(filepath, "rb") as fIn:
                cache_data = pickle.load(fIn)
                pub_text = cache_data["articles"]
                pub_vec = cache_data["embeddings"]
    
    embeddings[pub] = {
        "articles": pub_text,
        "embeddings": pub_vec
    }  
# %%
