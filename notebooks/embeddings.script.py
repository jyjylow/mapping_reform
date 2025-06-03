
__generated_with = "0.13.7"

# %%
import marimo as mo
import torch
import os
import pyarrow.parquet as pa

# text stuff
from wordcloud import WordCloud
from wordcloud import STOPWORDS
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
mo.md(r"""# functions""")

# %%
# reading in parquet files and converting them to df
def pa_to_df(parquet_path):
    df_raw = pa.read_table(parquet_path).to_pandas()
    return df_raw

# %%
# function to filter and simplify publication dfs
def clean_df(df_raw):
    # keeping only rows classified as 'text', excluding for e.g. titles
    df = df_raw.loc[df_raw['class'] == 'text']

    # keeping issue_id and content columns
    df = df[['issue_id', 'content']]

    # creating datetime col from issue_id
    df['pub_date'] = pd.to_datetime(df['issue_id'].str[-10:], format='%Y-%m-%d')

    return df

# %%
# function for EDA plots: (i) wordcloud, (ii) timeline distribution

# making a copy of wordcloud's default stopwords then adding to it
my_stopwords = set(STOPWORDS)
my_stopwords.update(["one", "will", "may", "time"])

def exploreplots(df):

    # setting up a 1-row, 2-column plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # extracting strings from 'content'
    df_list = df['content'].tolist()
    df_alltext = "".join(df_list)
    df_wordcloud = WordCloud(stopwords = my_stopwords).generate(df_alltext)

    # Plot (1): Word Cloud
    axes[0].imshow(df_wordcloud, interpolation='bilinear')
    axes[0].axis("off")
    axes[0].set_title("Word Cloud", fontsize=14)

    # Plot (2): Publication Date Histogram
    counts = df.groupby(df['pub_date'].dt.year).size()
    axes[1].bar(counts.index.astype(str), counts.values, color='skyblue')
    axes[1].set_title("Publication Date Distribution", fontsize=14)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Number of articles")

    plt.tight_layout()
    plt.show()

# %%
# reading in women's journal
ewj_raw = pa_to_df('data/post_processed/English_Womans_Journal_issue_PDF_files.parquet')
ewj = clean_df(ewj_raw)
ewj.head()

# %%
# reading in tomahawk
th_raw = pa_to_df('data/post_processed/Tomahawk_issue_PDF_files.parquet')
th = clean_df(th_raw)
th.head()

# %%
# reading in the leader
leader_raw = pa_to_df('data/post_processed/Leader_issue_PDF_files.parquet')
leader_raw.head()

# here i've found that there's a misformatted date, '1852-31-07'. fixing this below:
leader_raw.loc[leader_raw['issue_id'].str[-10:] == '1852-31-07', 'issue_id'] = '1852-07-31'

# cleaning
leader = clean_df(leader_raw)
leader.head()

# %%
exploreplots(ewj)

# %%
exploreplots(th)

# %%
exploreplots(leader)

# %%
mo.md(r"""## __Trying out static embedding__""")

# %%
mo.md(r"""## ncse embeddings""")

# %%
# downloading model from huggingface
# NOTE: specifying CPU here because M chip incompatibilities
model = SentenceTransformer("tomaarsen/static-retrieval-mrl-en-v1",device=torch.device('cpu'))

# %%
# extracting sample subset of ewj
ewj_test = ewj.sample(n=5, random_state = 123)
ewj_test_list = ewj_test["content"].tolist()
ewj_test_list

# %%
# getting vectors for test set
test_embed = model.encode(ewj_test_list)
print(test_embed.shape)
test_embed

# %%
mo.md(r"""## query embeddings""")

# %%
# testing queries. random for now
queries = ["women",
          'prison',
          'slavery',
          'law']

q_embed = model.encode(queries)
print(q_embed.shape)

# %%
# computing article-vector cosine similarity scores

cosine_similarity(test_embed,q_embed)