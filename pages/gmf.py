from dash import (
    html,
    Input,
    Output,
    State,
    callback,
    dcc
)

import dash
import dash_mantine_components as dmc

import plotly.graph_objects as go

import neurova
from neurova import io

import numpy as np
import os
import base64
import time

from functools import lru_cache

# =========================================================
# PAGE REGISTRATION
# =========================================================
dash.register_page(
    __name__,
    path="/gmf",
    name="GMF"
)

# =========================================================
# CONFIG
# =========================================================
MAX_IMAGES = 50
THUMBNAIL_WIDTH = "180px"

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif"
)

# =========================================================
# IMAGE HELPERS
# =========================================================
def image_to_base64(image_path):

    with open(image_path, "rb") as f:

        encoded = base64.b64encode(
            f.read()
        ).decode()

    extension = image_path.split(".")[-1]

    return (
        f"data:image/{extension};base64,{encoded}"
    )


def create_thumbnail_style():

    return {

        "width": "100%",
        "height": "180px",
        "objectFit": "cover",
        "borderRadius": "12px",
        "cursor": "pointer",
        "border": "2px solid #ddd",
        "transition": "0.2s"
    }


# =========================================================
# FEATURE EXTRACTION
# =========================================================
@lru_cache(maxsize=128)
def cached_image_read(path):

    image = io.imread(path)

    if len(image.shape) == 3:

        gray = np.mean(
            image,
            axis=2
        )

    else:

        gray = image

    return gray.astype(float)


def extract_all_features(img):

    features = {}

    features["mean"] = float(np.mean(img))
    features["std"] = float(np.std(img))
    features["variance"] = float(np.var(img))
    features["min"] = float(np.min(img))
    features["max"] = float(np.max(img))
    features["median"] = float(np.median(img))
    features["range"] = float(np.max(img) - np.min(img))

    features["p10"] = float(np.percentile(img, 10))
    features["p25"] = float(np.percentile(img, 25))
    features["p75"] = float(np.percentile(img, 75))
    features["p90"] = float(np.percentile(img, 90))

    features["iqr"] = float(
        np.percentile(img, 75)
        - np.percentile(img, 25)
    )

    features["energy"] = float(np.sum(img ** 2))

    features["rms"] = float(
        np.sqrt(np.mean(img ** 2))
    )

    # =========================
    # ENTROPY
    # =========================
    hist_entropy, _ = np.histogram(
        img,
        bins=64,
        range=(0, 256),
        density=True
    )

    hist_entropy = hist_entropy + 1e-12

    entropy = -np.sum(
        hist_entropy * np.log2(hist_entropy)
    )

    features["entropy"] = float(entropy)

    # =========================
    # HISTOGRAM
    # =========================
    hist, _ = np.histogram(
        img,
        bins=32,
        range=(0, 256)
    )

    normalized_hist = hist / hist.sum()

    features["histogram"] = (
        normalized_hist.tolist()
    )

    # =========================
    # GRADIENTS
    # =========================
    gx = np.diff(img, axis=1)
    gy = np.diff(img, axis=0)

    gradient_magnitude = np.sqrt(
        gx[:-1, :] ** 2
        + gy[:, :-1] ** 2
    )

    features["gradient_x"] = float(
        np.mean(np.abs(gx))
    )

    features["gradient_y"] = float(
        np.mean(np.abs(gy))
    )

    features["gradient_magnitude_mean"] = float(
        np.mean(gradient_magnitude)
    )

    features["gradient_magnitude_std"] = float(
        np.std(gradient_magnitude)
    )

    # =========================
    # PIXEL RATIOS
    # =========================
    total_pixels = img.size

    features["bright_pixel_ratio"] = float(
        np.sum(img > 200) / total_pixels
    )

    features["dark_pixel_ratio"] = float(
        np.sum(img < 50) / total_pixels
    )

    return features


# =========================================================
# PLOT HELPERS
# =========================================================
def create_histogram_figure(histogram_values):

    fig = go.Figure()

    fig.add_trace(

        go.Bar(

            x=[
                f"Bin {i}"
                for i in range(
                    len(histogram_values)
                )
            ],

            y=histogram_values
        )
    )

    fig.update_layout(

        title="Histogram",

        height=320,

        template="plotly_white",

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        )
    )

    return fig


def create_distribution_curve(img):

    flattened = img.flatten()

    hist, bins = np.histogram(
        flattened,
        bins=32,
        range=(0, 256),
        density=True
    )

    centers = (
        bins[:-1] + bins[1:]
    ) / 2

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=centers,
            y=hist,

            mode="lines",

            fill="tozeroy"
        )
    )

    fig.update_layout(

        title="Distribution Curve",

        height=320,

        template="plotly_white"
    )

    return fig


def create_heatmap(img):

    # =====================================
    # DOWNSAMPLE MASSIVELY
    # =====================================
    reduced = img[::8, ::8]

    fig = go.Figure(

        data=go.Heatmap(
            z=reduced
        )
    )

    fig.update_layout(

        title="Heatmap",

        height=420,

        template="plotly_white"
    )

    return fig


def create_box_plot(img):

    sampled = img.flatten()[::20]

    fig = go.Figure()

    fig.add_trace(

        go.Box(
            y=sampled,
            boxpoints=False
        )
    )

    fig.update_layout(

        title="Box Plot",

        height=320,

        template="plotly_white"
    )

    return fig


# =========================================================
# LAYOUT
# =========================================================
layout = dmc.MantineProvider(

    dmc.Container(

        [

            dmc.Title(
                "Advanced Image Feature Visualizer",
                order=2,
                mb="md"
            ),

            dmc.Text(
                (
                    "Fast lazy-loading analytics "
                    "for image folders."
                ),
                c="dimmed",
                mb="lg"
            ),

            dmc.TextInput(
                id="input-folder-path",
                placeholder="Enter folder path",
                label="Folder Path",
                w=500
            ),

            dmc.Space(h=20),

            dmc.Button(
                "Load Folder",
                id="load-folder-btn",
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(
                id="gallery-output"
            ),

            dmc.Space(h=40),

            html.Div(
                id="selected-image-output"
            )

        ],

        fluid=True
    )
)

# =========================================================
# LOAD GALLERY
# =========================================================
@callback(

    Output(
        "gallery-output",
        "children"
    ),

    Input(
        "load-folder-btn",
        "n_clicks"
    ),

    State(
        "input-folder-path",
        "value"
    )
)
def load_gallery(
    n_clicks,
    folder_path
):

    if not n_clicks or not folder_path:

        return ""

    try:

        image_files = [

            file

            for file in os.listdir(
                folder_path
            )

            if file.lower().endswith(
                SUPPORTED_EXTENSIONS
            )
        ]

        image_files = image_files[:MAX_IMAGES]

        if not image_files:

            return dmc.Alert(
                "No images found.",
                color="yellow"
            )

        cards = []

        for file_name in image_files:

            full_path = os.path.join(
                folder_path,
                file_name
            )

            img = html.Img(

                src=image_to_base64(
                    full_path
                ),

                style=create_thumbnail_style(),

                id={
                    "type": "image-thumb",
                    "index": full_path
                }
            )

            card = dmc.Card(

                [

                    img,

                    dmc.Space(h=10),

                    dmc.Text(
                        file_name,
                        size="sm",
                        truncate=True
                    )

                ],

                shadow="sm",
                radius="md",
                withBorder=True
            )

            cards.append(card)

        return dmc.SimpleGrid(
            cols=5,
            spacing="lg",
            children=cards
        )

    except Exception as e:

        return dmc.Alert(
            str(e),
            color="red"
        )


# =========================================================
# DISPLAY SINGLE IMAGE ANALYTICS
# =========================================================
@callback(

    Output(
        "selected-image-output",
        "children"
    ),

    Input(
        {
            "type": "image-thumb",
            "index": dash.ALL
        },
        "n_clicks"
    ),

    prevent_initial_call=True
)
def show_image_analytics(clicks):

    ctx = dash.callback_context

    if not ctx.triggered:

        return ""

    triggered_id = ctx.triggered_id

    image_path = triggered_id["index"]

    try:

        start = time.time()

        gray_image = cached_image_read(
            image_path
        )

        features = extract_all_features(
            gray_image
        )

        # =====================================
        # FIGURES
        # =====================================
        histogram_fig = (
            create_histogram_figure(
                features["histogram"]
            )
        )

        distribution_fig = (
            create_distribution_curve(
                gray_image
            )
        )

        heatmap_fig = create_heatmap(
            gray_image
        )

        box_plot_fig = create_box_plot(
            gray_image
        )

        elapsed = round(
            time.time() - start,
            2
        )

        # =====================================
        # FEATURE CARDS
        # =====================================
        feature_cards = []

        for k, v in features.items():

            if k == "histogram":

                continue

            feature_cards.append(

                dmc.Paper(

                    [

                        dmc.Text(
                            k,
                            fw=700,
                            size="sm"
                        ),

                        dmc.Text(

                            (
                                f"{v:.4f}"

                                if isinstance(
                                    v,
                                    (int, float)
                                )

                                else str(v)
                            ),

                            size="xs",
                            c="dimmed"
                        )

                    ],

                    p="sm",

                    radius="md",

                    shadow="xs",

                    withBorder=True
                )
            )

        analytics = dmc.Paper(

            [

                dmc.Group(

                    [

                        dmc.Title(
                            os.path.basename(
                                image_path
                            ),
                            order=3
                        ),

                        dmc.Badge(
                            f"{elapsed}s",
                            color="green"
                        )

                    ],

                    justify="space-between",
                    mb="lg"
                ),

                html.Img(

                    src=image_to_base64(
                        image_path
                    ),

                    style={

                        "width": "100%",

                        "maxWidth": "500px",

                        "borderRadius": "12px",

                        "marginBottom": "20px"
                    }
                ),

                dmc.Title(
                    "Features",
                    order=4,
                    mb="md"
                ),

                dmc.SimpleGrid(
                    cols=4,
                    spacing="md",
                    children=feature_cards
                ),

                dmc.Space(h=30),

                dmc.Accordion(

                    chevronPosition="right",

                    variant="separated",

                    children=[

                        dmc.AccordionItem(

                            [

                                dmc.AccordionControl(
                                    "Histogram"
                                ),

                                dmc.AccordionPanel(

                                    dcc.Graph(
                                        figure=histogram_fig
                                    )
                                )

                            ],

                            value="histogram"
                        ),

                        dmc.AccordionItem(

                            [

                                dmc.AccordionControl(
                                    "Distribution Curve"
                                ),

                                dmc.AccordionPanel(

                                    dcc.Graph(
                                        figure=distribution_fig
                                    )
                                )

                            ],

                            value="distribution"
                        ),

                        dmc.AccordionItem(

                            [

                                dmc.AccordionControl(
                                    "Heatmap"
                                ),

                                dmc.AccordionPanel(

                                    dcc.Graph(
                                        figure=heatmap_fig
                                    )
                                )

                            ],

                            value="heatmap"
                        ),

                        dmc.AccordionItem(

                            [

                                dmc.AccordionControl(
                                    "Box Plot"
                                ),

                                dmc.AccordionPanel(

                                    dcc.Graph(
                                        figure=box_plot_fig
                                    )
                                )

                            ],

                            value="boxplot"
                        )

                    ]
                )

            ],

            p="xl",

            radius="lg",

            shadow="md",

            withBorder=True,

            mb="xl"
        )

        return analytics

    except Exception as e:

        return dmc.Alert(
            str(e),
            color="red",
            title="Processing Failed"
        )
