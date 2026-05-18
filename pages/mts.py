import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import os
import nltk
import dash

from dash import (
    html,
    Input,
    Output,
    callback,
    ctx
)

import dash_mantine_components as dmc

from txtai import LLM
from txtai import Textractor
from txtai import Embeddings


# ====================================================
# Setup
# ====================================================

nltk.download("punkt", quiet=True)

SUPPORTED_FORMATS = ("docx", "xlsx", "pdf")


# ====================================================
# Load models ONCE
# ====================================================

# print("=" * 60)
# print("Loading LLM...")
# print("=" * 60)

llm = LLM(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

# print("LLM loaded")


# print("=" * 60)
# print("Loading Textractor...")
# print("=" * 60)

textractor = Textractor(
    paragraphs=True
)

# print("Textractor loaded")


# print("=" * 60)
# print("Loading Embeddings...")
# print("=" * 60)

embeddings = Embeddings(
    content=True,
    path="sentence-transformers/all-MiniLM-L6-v2"
)

# print("Embeddings loaded")


# ====================================================
# Dash page
# ====================================================

dash.register_page(
    __name__,
    path="/mts",
    name="MTS"
)


# ====================================================
# Layout
# ====================================================

layout = dmc.MantineProvider(

    dmc.Container([

        dmc.Title(
            "Single-source Text Search",
            order=2
        ),

        dmc.Space(h=30),

        dmc.TextInput(
            id="input-file-path-mts",
            label="File Path",
            placeholder="Enter file path",
            w=600
        ),

        dmc.Space(h=20),

        dmc.TextInput(
            id="input-search-query-mts",
            label="Search Query",
            placeholder="Enter query",
            w=600
        ),

        dmc.Space(h=20),

        dmc.Button(
            "Search Document",
            id="submit-button-mts",
            n_clicks=0
        ),

        dmc.Space(h=30),

        html.Div(
            id="output-message-mts"
        )

    ])
)


# ====================================================
# Document extraction
# ====================================================

def stream(file_path):

    try:

        extension = file_path.split(".")[-1].lower()

        if extension not in SUPPORTED_FORMATS:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        # print("=" * 60)
        # print(f"Indexing: {file_path}")

        count = 0

        for paragraph in textractor(
            file_path
        ):

            if paragraph and str(paragraph).strip():

                count += 1

                # print(
                #     f"Paragraph {count}"
                # )

                yield paragraph

        # print(
        #     f"Finished processing file"
        # )

        # print(
        #     f"Total paragraphs: {count}"
        # )

    except Exception as e:

        # print(
        #     "Document extraction error:"
        # )

        # print(
        #     str(e)
        # )
        pass

# ====================================================
# Context Retrieval
# ====================================================

def get_context(question):

    # print("=" * 60)
    # print("Searching embeddings...")
    # print("=" * 60)

    results = embeddings.search(
        question,
        limit=3
    )

    if not results:

        return "No matching context found"

    context = "\n".join(

        x["text"]

        for x in results
        if "text" in x
    )

    context = context[:2000]

    # print("=" * 60)
    # print("Context Preview:")
    # print(context[:500])
    # print("=" * 60)

    return context


# ====================================================
# LLM execution
# ====================================================

def execute(question, context):

    try:

        # print("=" * 60)
        # print("Sending prompt to LLM...")
        # print("=" * 60)

        response = llm(

            [
                {
                    "role": "system",
                    "content":
                    "Answer only using supplied context."
                },

                {
                    "role": "user",
                    "content":
                    f"""
Question:
{question}

Context:
{context}
"""
                }

            ],

            maxlength=256,
            # maxnew=128
        )

        # print("=" * 60)
        # print("LLM finished")
        # print("=" * 60)

        return response

    except Exception as e:

        # print("LLM ERROR:")
        # print(str(e))

        return f"LLM Error: {str(e)}"


def rag(question):

    context = get_context(question)

    return execute(
        question,
        context
    )


# ====================================================
# Callback
# ====================================================

@callback(

    Output(
        "output-message-mts",
        "children"
    ),

    Input(
        "input-file-path-mts",
        "value"
    ),

    Input(
        "submit-button-mts",
        "n_clicks"
    ),

    Input(
        "input-search-query-mts",
        "value"
    ),

    prevent_initial_call=True
)

def update_file_path_mts(
    file_path,
    clicks,
    query
):

    try:

        triggered_id = ctx.triggered_id

        # print(
        #     f"Triggered by: {triggered_id}"
        # )

        if triggered_id != "submit-button-mts":
            return dash.no_update

        if not file_path:

            return [
                dmc.Text(
                    "Please enter a file path"
                )
            ]

        if not query:

            return [
                dmc.Text(
                    "Please enter a search query"
                )
            ]

        if not os.path.isfile(file_path):

            return [
                dmc.Text(
                    f"File not found: {file_path}"
                )
            ]

        # print("=" * 60)
        # print("Starting indexing...")
        # print("=" * 60)

        global embeddings

        embeddings = Embeddings(
            content=True,
            path="sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings.index(
            stream(file_path)
        )

        # print("=" * 60)
        # print("Indexing complete")
        # print("=" * 60)

        answer = rag(query)

        return [

            dmc.Text(
                "Document indexed successfully"
            ),

            dmc.Space(h=20),

            dmc.Text(
                "Answer:",
                fw=700
            ),

            dmc.Space(h=10),

            dmc.Text(
                str(answer)
            )
        ]

    except Exception as e:

        # print("=" * 60)
        # print("APPLICATION ERROR:")
        # print(str(e))
        # print("=" * 60)

        return [
            dmc.Text(
                f"Error: {str(e)}"
            )
        ]