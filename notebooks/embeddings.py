import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import pyarrow.parquet as pa
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import StaticEmbedding
    from tokenizers import Tokenizer
    return SentenceTransformer, mo, pa, pd


@app.cell
def _(pa):
    # reading in women's journal
    ewj_raw = pa.read_table('data/post_processed/English_Womans_Journal_issue_PDF_files.parquet').to_pandas()
    ewj_raw.head()
    return (ewj_raw,)


@app.cell
def _(ewj_raw, pd):
    # cleaning ewj dataset

    # 1. filtering ewj dataset to "text" rows
    ewj = ewj_raw.loc[ewj_raw['class'] == 'text']

    # 2. dropping columns i don't need
    ewj = ewj[['issue_id', 'content']]

    #3. converting issue_id to datetime format
    ewj['pub_date'] = pd.to_datetime(ewj['issue_id'].str[-10:], format='%Y-%m-%d')
    ewj.head()
    return (ewj,)


@app.cell
def _(mo):
    mo.md(r"""## __Trying out static embedding__""")
    return


@app.cell
def _(SentenceTransformer):
    # downloading model from huggingface
    model = SentenceTransformer("tomaarsen/static-retrieval-mrl-en-v1")

    return (model,)


@app.cell
def _(ewj):
    # extracting sample subset of ewj
    ewj_test = ewj.sample(n=5, random_state = 123)
    ewj_test_list = ewj_test["content"].tolist()
    ewj_test_list
    return (ewj_test_list,)


@app.cell
def _(embeddings, ewj_test_list, model):
    embeddings_test = model.encode(ewj_test_list)
    print(embeddings.shape)
    return


if __name__ == "__main__":
    app.run()
