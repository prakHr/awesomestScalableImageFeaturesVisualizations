from dash import html, Input, Output, State, callback
import dash

import dash_mantine_components as dmc
import neurova
from neurova import io
import numpy as np


import os

dash.register_page(
    __name__,
    path="/gmf",
    name="GMF"
)


def extract_all_features(img):

    features = {}

    # Statistical features
    features["mean"] = np.mean(img)
    features["std"] = np.std(img)
    features["min"] = np.min(img)
    features["max"] = np.max(img)
    features["median"] = np.median(img)

    # Histogram features
    hist, _ = np.histogram(
        img,
        bins=16,
        range=(0, 256)
    )

    features["histogram"] = (
        hist / hist.sum()
    ).tolist()

    # Gradient features
    gx = np.diff(img.astype(float), axis=1)
    gy = np.diff(img.astype(float), axis=0)

    features["gradient_x"] = np.mean(np.abs(gx))
    features["gradient_y"] = np.mean(np.abs(gy))

    return features


layout = dmc.MantineProvider(

    dmc.Container(

        [

            dmc.Title(
                "Folder Image Feature Extractor",
                order=2,
                mb="md"
            ),

            dmc.TextInput(
                id='input-folder-path',
                placeholder='Enter folder path',
                label="Folder Path",
                w=500
            ),

            dmc.Space(h=20),

            dmc.Button(
                "Extract Features",
                id='submit-folder-button',
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(id='folder-output')

        ]
    )
)


@callback(
    Output('folder-output', 'children'),
    Input('submit-folder-button', 'n_clicks'),
    State('input-folder-path', 'value')
)
def display_folder_features(n_clicks, folder_path):

    if not n_clicks or not folder_path:
        return ""

    try:

        supported_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif"
        )

        image_files = [

            file for file in os.listdir(folder_path)

            if file.lower().endswith(
                supported_extensions
            )
        ]

        if not image_files:

            return dmc.Alert(
                "No images found in folder.",
                color="yellow",
                title="Empty Folder"
            )

        all_components = []

        for file_name in image_files:

            full_path = os.path.join(
                folder_path,
                file_name
            )

            try:

                image = io.imread(full_path)

                features = extract_all_features(image)

                feature_cards = dmc.SimpleGrid(

                    cols=2,

                    children=[

                        dmc.Paper(

                            [

                                dmc.Text(
                                    str(k),
                                    fw=700,
                                    size="sm"
                                ),

                                dmc.Text(
                                    str(v),
                                    c="dimmed",
                                    size="xs"
                                )

                            ],

                            p="sm",
                            radius="md",
                            shadow="xs",
                            withBorder=True
                        )

                        for k, v in features.items()
                    ]
                )

                image_section = dmc.Paper(

                    [

                        dmc.Title(
                            file_name,
                            order=4,
                            mb="md"
                        ),

                        feature_cards

                    ],

                    p="lg",
                    shadow="md",
                    radius="lg",
                    withBorder=True,
                    mb="xl"
                )

                all_components.append(
                    image_section
                )

            except Exception as image_error:

                all_components.append(

                    dmc.Alert(
                        f"{file_name}: {str(image_error)}",
                        color="red",
                        title="Image Processing Failed"
                    )
                )

        return all_components

    except Exception as e:

        return dmc.Alert(
            f"Error: {str(e)}",
            color="red",
            title="Folder Loading Failed"
        )