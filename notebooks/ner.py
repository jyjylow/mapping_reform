import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import pyarrow.parquet as pa
    from gliner import GLiNER
    return GLiNER, mo, pa, pd


@app.cell
def _(pa):
    # reading in Tomahawk parquet file using pyArrow and converting to Pandas dataframe

    tomahawk = pa.read_table('data/post_processed/Tomahawk_issue_PDF_files.parquet').to_pandas()
    tomahawk.head()
    return (tomahawk,)


@app.cell
def _(tomahawk):
    # taking a look at columns and data types
    tomahawk.info()
    return


@app.cell
def _(tomahawk):
    # changing 'class' column from string to categorical
    tomahawk['class'] = tomahawk['class'].astype('category')
    return


@app.cell
def _(tomahawk):
    tomahawk['class'].value_counts()
    return


@app.cell
def _(pd, tomahawk):
    # cleaning tomahawk dataset

    # 1. filtering tomahawk dataset to "text" rows
    thawk = tomahawk.loc[tomahawk['class'] == 'text']

    # 2. dropping columns i don't need
    thawk = thawk[['issue_id', 'content']]

    #3. converting issue_id to datetime format
    thawk['pub_date'] = pd.to_datetime(thawk['issue_id'].str[-10:], format='%Y-%m-%d')
    return (thawk,)


@app.cell
def _(thawk):
    thawk.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## __Importing GLiNER model (pretrained, fine-tuned)__""")
    return


@app.cell
def _(GLiNER):
    # trying with EmergentMethod's fine-tuned model
    model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")

    return (model,)


@app.cell
def _():
    # identifying labels to extract
    labels = ["person", "location", "date", "event", "organization", "place", "nationality", "language", "quantity", "title", "conflict"]
    return (labels,)


@app.cell
def _(mo):
    mo.md(r"""### __Function for entity extraction__""")
    return


@app.function
# defining function for entity extraction
def extract_entities(text, labels, model):
    entities = model.predict_entities(text=text, labels=labels)
    
    # initiating empty dictionary to store output
    results_dict = {label: [] for label in labels}

    # adding extracted entities to output dict
    for entity in entities:
        results_dict[entity['label']].append(entity["text"])

    # joining outputs as comma-separated strings, or empty strings if no identified entities for that label
    entity_summary = {label: ", ".join(results_dict[label]) if results_dict[label] else "" for label in labels}

    return entity_summary


@app.cell
def _(mo):
    mo.md(r"""### __TEST__""")
    return


@app.cell
def _(tomahawk):
    test=tomahawk.sample(n=1, random_state=1225)
    testtext= test["content"].iloc[0]
    testtext
    return (testtext,)


@app.cell
def _(labels, model, testtext):
    extract_entities(testtext, labels, model)
    return


@app.cell
def _(mo):
    mo.md(r"""### __Working on subset__""")
    return


@app.cell
def _(labels, model, thawk):
    for text in thawk['content'].head(n=10):
        results = extract_entities(text, labels, model)
        print(results)
    return


if __name__ == "__main__":
    app.run()
