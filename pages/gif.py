from dash import html, Input, Output, State, callback
import dash

from dash import Dash, Input, Output, State, html
import dash_mantine_components as dmc

import neurova as nv

import base64
import matplotlib.pyplot as plt
import io as pyio

from neurova import io, transform, core, datasets

import numpy as np

dash.register_page(
    __name__,
    path="/gif",
    name="GIF"
)


def extract_all_features(img):
    """Extract comprehensive feature vector from image."""
    features = {}
    
# statistical features
    # features.extend([
        # np.mean(img), np.std(img), np.min(img), np.max(img), np.median(img)
    # ])
    features["mean"] = np.mean(img)
    features["std"] = np.std(img)
    features["min"] = np.min(img)
    features["max"] = np.max(img)
    features["median"] = np.std(img)
    
    # Histogram features (binned)
    hist, _ = np.histogram(img, bins=16, range=(0, 256))
    features["histogram features"] = hist / hist.sum()
    
    # features.extend(hist / hist.sum())
    
    # gradient features
    gx = np.diff(img.astype(float), axis=1)
    gy = np.diff(img.astype(float), axis=0)
    features["gradientx features"] = np.mean(np.abs(gx))
    features["gradienty features"] = np.mean(np.abs(gy))
    
    # features.extend([np.mean(np.abs(gx)), np.mean(np.abs(gy))])
    
    return features


layout = dmc.MantineProvider(

    dmc.Container(

        [

            dmc.Title(
                "Dash Image Features Extractor Viewer",
                order=2,
                mb="md"
            ),

            dmc.TextInput(
                id='input-text21',
                placeholder='Enter image path',
                label="Image Path",
                w=400
            ),

            dmc.Space(h=20),

            dmc.Button(
                "Click To See Image Features",
                id='submit-button21',
                n_clicks=0
            ),

            dmc.Space(h=20),

            html.Div(id='output-message21'),
            
        ]

    )
)



@callback(
    Output('output-message21', 'children'),
    Input('submit-button21', 'n_clicks'),
    State('input-text21', 'value')
)
def display_image_features(n_clicks, input_value):

    if n_clicks > 0 and input_value:

        try:

            # Read image
            image = io.imread(input_value)

            data = extract_all_features(image)

            return [dmc.SimpleGrid(

                cols=2,

                children=[

                    dmc.Paper(

                        [
                            dmc.Text(
                                str(k),
                                fw=700,
                                size="lg"
                            ),

                            dmc.Text(
                                str(v),
                                c="dimmed"
                            )
                        ],

                        p="md",
                        radius="md",
                        shadow="sm",
                        withBorder=True
                    )

                    for k, v in data.items()
                ]
            )
        ]
        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""

