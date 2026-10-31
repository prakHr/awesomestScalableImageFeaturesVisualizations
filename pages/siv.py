import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

from dash import html, Input, Output, State, callback
import dash
import dash_mantine_components as dmc

import neurova as nv
from neurova import io, transform, core

import numpy as np
import img2sktch as i2s

import matplotlib.pyplot as plt
import base64
import io as pyio


dash.register_page(
    __name__,
    path="/siv",
    name="SIV"
)


# -----------------------------
# Utility Functions
# -----------------------------

def normalize_image(arr):
    """Normalize image for matplotlib display"""

    arr = np.array(arr)

    if arr.dtype != np.uint8:

        arr = arr.astype(np.float32)

        min_val = arr.min()
        max_val = arr.max()

        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)

        arr = np.clip(arr, 0, 1)

    return arr


def create_image_component(arr):

    fig = None

    try:

        arr = normalize_image(arr)

        fig, ax = plt.subplots(figsize=(6, 6))

        if len(arr.shape) == 2:
            ax.imshow(arr, cmap='gray')
        else:
            ax.imshow(arr)

        ax.axis("off")

        buf = pyio.BytesIO()

        plt.savefig(
            buf,
            format="png",
            bbox_inches='tight',
            pad_inches=0
        )

        buf.seek(0)

        img_base64 = base64.b64encode(
            buf.read()
        ).decode("utf-8")

        return dmc.Paper(
            [
                dmc.Image(
                    src=f"data:image/png;base64,{img_base64}",
                    w=400,
                    radius="md"
                )
            ],
            shadow="sm",
            radius="md",
            p="md",
            withBorder=True
        )

    finally:
        if fig:
            plt.close(fig)


def error_component(e):

    return dmc.Alert(
        f"Error: {str(e)}",
        color="red",
        title="Operation Failed"
    )


# -----------------------------
# Layout
# -----------------------------

layout = dmc.MantineProvider(

    dmc.Container(

        [

            dmc.Title(
                "Dash Image Viewer",
                order=2,
                mb="md"
            ),

            dmc.TextInput(
                id='input-text',
                placeholder='Enter image path',
                label="Image Path",
                w=400
            ),

            dmc.Space(h=20),

            dmc.TextInput(
                id='input-sketch-text',
                placeholder='Enter blur value',
                label="Blur value",
                w=400
            ),

            dmc.Button(
                "Sketch",
                id='submit-sketch-button'
            ),

            html.Div(id='output-sketch-message'),

            dmc.Space(h=20),

            dmc.Button(
                "Grayscale",
                id='submit-button'
            ),

            html.Div(id='output-message'),

            dmc.Space(h=20),

            dmc.Button(
                "HSV",
                id='submit-button2'
            ),

            html.Div(id='output-message2'),

            dmc.Space(h=20),

            dmc.Button(
                "RGB",
                id='submit-button3'
            ),

            html.Div(id='output-message3'),

            dmc.Space(h=20),

            dmc.TextInput(
                id='input-resize-h-text',
                label="Height"
            ),

            dmc.TextInput(
                id='input-resize-w-text',
                label="Width"
            ),

            dmc.Button(
                "Resize",
                id='submit-button4'
            ),

            html.Div(id='output-message4'),

            dmc.Space(h=20),

            dmc.TextInput(
                id='input-rotation-text',
                label="Rotation Angle"
            ),

            dmc.Button(
                "Rotate",
                id='submit-button5'
            ),

            html.Div(id='output-message5'),

            dmc.Space(h=20),

            dmc.Button(
                "Flip",
                id='submit-button6'
            ),

            html.Div(id='output-message6'),

        ],

        size="md",
        pt=40
    )
)


# -----------------------------
# Sketch
# -----------------------------

@callback(
    Output("output-sketch-message", "children"),
    Input("submit-sketch-button", "n_clicks"),
    State("input-text", "value"),
    State("input-sketch-text", "value")
)
def display_sketch(n, path, blur):

    if not n:
        return ""

    try:

        image = i2s.from_file(path)

        arr = i2s.PencilSketch(
            blur_sigma=int(blur)
        )(image)

        return create_image_component(arr)

    except Exception as e:
        return error_component(e)


# -----------------------------
# Grayscale
# -----------------------------

@callback(
    Output("output-message", "children"),
    Input("submit-button", "n_clicks"),
    State("input-text", "value")
)
def display_grayscale(n, path):

    if not n:
        return ""

    try:

        image = io.imread(path)

        arr = core.to_grayscale(image)

        return create_image_component(arr)

    except Exception as e:
        return error_component(e)


# -----------------------------
# HSV
# -----------------------------

@callback(
    Output("output-message2", "children"),
    Input("submit-button2", "n_clicks"),
    State("input-text", "value")
)
def display_hsv(n, path):

    if not n:
        return ""

    try:

        image = io.imread(path)

        arr = core.convert_color_space(
            image,
            core.ColorSpace.HSV,
            from_space=core.ColorSpace.RGB
        )

        return create_image_component(arr)

    except Exception as e:
        return error_component(e)


# -----------------------------
# RGB
# -----------------------------

@callback(
    Output("output-message3", "children"),
    Input("submit-button3", "n_clicks"),
    State("input-text", "value")
)
def display_rgb(n, path):

    if not n:
        return ""

    try:

        arr = io.imread(path)

        return create_image_component(arr)

    except Exception as e:
        return error_component(e)


# -----------------------------
# Resize
# -----------------------------

@callback(
    Output("output-message4", "children"),
    Input("submit-button4", "n_clicks"),
    State("input-text", "value"),
    State("input-resize-h-text", "value"),
    State("input-resize-w-text", "value")
)
def resize_image(n, path, h, w):

    if not n:
        return ""

    try:

        arr = io.imread(path)

        arr = transform.resize(
            arr,
            (int(h), int(w))
        )

        return create_image_component(arr)

    except Exception as e:
        return error_component(e)


# -----------------------------
# Rotate
# -----------------------------

@callback(
    Output("output-message5", "children"),
    Input("submit-button5", "n_clicks"),
    State("input-text", "value"),
    State("input-rotation-text", "value")
)
def rotate_image(n, path, angle):

    if not n:
        return ""

    try:

        arr = io.imread(path)

        arr = transform.rotate(
            arr,
            int(angle)
        )

        return create_image_component(arr)

    except Exception as e:
        return error_component(e)


# -----------------------------
# Flip
# -----------------------------

@callback(
    Output("output-message6", "children"),
    Input("submit-button6", "n_clicks"),
    State("input-text", "value")
)
def flip_image(n, path):

    if not n:
        return ""

    try:

        image = io.imread(path)

        flipped_images = [

            np.flip(image, axis=1),
            np.flip(image, axis=0),
            np.flip(np.flip(image, axis=0), axis=1)

        ]

        return [

            create_image_component(img)

            for img in flipped_images
        ]

    except Exception as e:
        return error_component(e)