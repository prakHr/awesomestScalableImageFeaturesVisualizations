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
    path="/siv",
    name="SIV"
)


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

            dmc.Button(
                "Click To See Grayscale Image",
                id='submit-button',
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(id='output-message'),
            
            dmc.Space(h=20),

            dmc.Button(
                "Click To See HSV Image",
                id='submit-button2',
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(id='output-message2'),

            dmc.Space(h=20),

            dmc.Button(
                "Click To See RGB Image",
                id='submit-button3',
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(id='output-message3'),

            dmc.TextInput(
                id='input-resize-h-text',
                placeholder='Enter height of image',
                label="For a Image Path",
                w=400
            ),

            dmc.TextInput(
                id='input-resize-w-text',
                placeholder='Enter width of image',
                label="For a Image Path",
                w=400
            ),

            dmc.Space(h=20),

            dmc.Button(
                "Click To See Resized Image",
                id='submit-button4',
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(id='output-message4'),

            dmc.TextInput(
                id='input-rotation-text',
                placeholder='Enter rotation angle of image',
                label="For a Image Path",
                w=400
            ),

            dmc.Space(h=20),

            html.Div(id='output-message5'),
            
            dmc.Space(h=30),

            dmc.Button(
                "Click To See Rotated Image",
                id='submit-button5',
                n_clicks=0
            ),

            dmc.Space(h=20),

            dmc.Button(
                "Click To See Flipped Image",
                id='submit-button6',
                n_clicks=0
            ),

            dmc.Space(h=30),

            html.Div(id='output-message6'),






        ],

        size="md",
        pt=40

    )

)

@callback(
    Output('output-message', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def display_grayscale_image(n_clicks, input_value):

    if n_clicks > 0 and input_value:

        try:

            # Read image
            image = io.imread(input_value)

            # Convert to grayscale
            arr = core.to_grayscale(image)

            # Create figure
            fig, ax = plt.subplots()

            ax.imshow(arr, cmap='gray')
            ax.axis('off')

            # Save to memory
            buf = pyio.BytesIO()

            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0
            )

            buf.seek(0)

            # Convert image to base64
            img_base64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            plt.close(fig)

            return dmc.Paper(

                [
                    dmc.Image(
                        src=f'data:image/png;base64,{img_base64}',
                        w=400,
                        radius="md"
                    )
                ],

                shadow="sm",
                radius="md",
                p="md",
                withBorder=True
            )
        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""

@callback(
    Output('output-message2', 'children'),
    Input('submit-button2', 'n_clicks'),
    State('input-text', 'value')
)
def display_hsv_image(n_clicks, input_value):

    if n_clicks > 0 and input_value:

        try:

            # Read image
            image = io.imread(input_value)

            arr = core.convert_color_space(image, core.ColorSpace.HSV, from_space=core.ColorSpace.RGB)

            # Create figure
            fig, ax = plt.subplots()

            ax.imshow(arr, cmap='gray')
            ax.axis('off')

            # Save to memory
            buf = pyio.BytesIO()

            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0
            )

            buf.seek(0)

            # Convert image to base64
            img_base64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            plt.close(fig)

            return dmc.Paper(

                [
                    dmc.Image(
                        src=f'data:image/png;base64,{img_base64}',
                        w=400,
                        radius="md"
                    )
                ],

                shadow="sm",
                radius="md",
                p="md",
                withBorder=True
            )
        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""

@callback(
    Output('output-message3', 'children'),
    Input('submit-button3', 'n_clicks'),
    State('input-text', 'value')
)
def display_rgb_image(n_clicks, input_value):

    if n_clicks > 0 and input_value:

        try:

            # Read image
            arr = io.imread(input_value)

            # Create figure
            fig, ax = plt.subplots()

            ax.imshow(arr, cmap='gray')
            ax.axis('off')

            # Save to memory
            buf = pyio.BytesIO()

            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0
            )

            buf.seek(0)

            # Convert image to base64
            img_base64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            plt.close(fig)

            return dmc.Paper(

                [
                    dmc.Image(
                        src=f'data:image/png;base64,{img_base64}',
                        w=400,
                        radius="md"
                    )
                ],

                shadow="sm",
                radius="md",
                p="md",
                withBorder=True
            )

        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""

@callback(
    Output('output-message4', 'children'),
    Input('submit-button4', 'n_clicks'),
    State('input-text', 'value'),
    State('input-resize-w-text', 'value'),
    State('input-resize-h-text', 'value')
)
def display_rgb_image(n_clicks, input_value, input_value1, input_value2):

    if n_clicks > 0 and input_value and input_value1 and input_value2:

        try:

            # Read image
            arr = io.imread(input_value)
            arr = transform.resize(arr, (int(input_value1), int(input_value2)))

            # Create figure
            fig, ax = plt.subplots()

            ax.imshow(arr, cmap='gray')
            ax.axis('off')

            # Save to memory
            buf = pyio.BytesIO()

            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0
            )

            buf.seek(0)

            # Convert image to base64
            img_base64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            plt.close(fig)

            return dmc.Paper(

                [
                    dmc.Image(
                        src=f'data:image/png;base64,{img_base64}',
                        w=400,
                        radius="md"
                    )
                ],

                shadow="sm",
                radius="md",
                p="md",
                withBorder=True
            )
        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""

@callback(
    Output('output-message5', 'children'),
    Input('submit-button5', 'n_clicks'),
    State('input-text', 'value'),
    State('input-rotation-text', 'value')
)
def display_rotated_image(n_clicks, input_value, input_value1):

    if n_clicks > 0 and input_value and input_value1:

        try:

            # Read image
            arr = io.imread(input_value)
            arr = transform.rotate(arr, int(input_value1))

            # Create figure
            fig, ax = plt.subplots()

            ax.imshow(arr, cmap='gray')
            ax.axis('off')

            # Save to memory
            buf = pyio.BytesIO()

            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0
            )

            buf.seek(0)

            # Convert image to base64
            img_base64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            plt.close(fig)

            return dmc.Paper(

                [
                    dmc.Image(
                        src=f'data:image/png;base64,{img_base64}',
                        w=400,
                        radius="md"
                    )
                ],

                shadow="sm",
                radius="md",
                p="md",
                withBorder=True
            )

        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""

@callback(
    Output('output-message6', 'children'),
    Input('submit-button6', 'n_clicks'),
    State('input-text', 'value')
)
def display_flipped_image(n_clicks, input_value):

    if n_clicks > 0 and input_value:

        try:

            # Read image
            rgb_image = io.imread(input_value)
            # Horizontal flip (mirror) - use numpy
            arr1 = np.flip(rgb_image, axis=1)
            # print(f"    Horizontal flip: {flipped_h.shape}")

            # vertical flip
            arr2 = np.flip(rgb_image, axis=0)
            # print(f"    Vertical flip: {flipped_v.shape}")

            # both axes
            arr3 = np.flip(np.flip(rgb_image, axis=0), axis=1)
            # print(f"    Both axes flip: {flipped_both.shape}")
            # arr = transform.rotate(arr, int(input_value1))
            arr_elements = []
            for arr in [arr1,arr2,arr3]:
                # Create figure
                fig, ax = plt.subplots()

                ax.imshow(arr, cmap='gray')
                ax.axis('off')

                # Save to memory
                buf = pyio.BytesIO()

                plt.savefig(
                    buf,
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0
                )

                buf.seek(0)

                # Convert image to base64
                img_base64 = base64.b64encode(
                    buf.read()
                ).decode('utf-8')

                plt.close(fig)

                arr_elements.append(
                 dmc.Paper(

                        [
                            dmc.Image(
                                src=f'data:image/png;base64,{img_base64}',
                                w=400,
                                radius="md"
                            )
                        ],

                        shadow="sm",
                        radius="md",
                        p="md",
                        withBorder=True
                    )
                )
            return arr_elements

        except Exception as e:

            return dmc.Alert(
                f"Error: {str(e)}",
                color="red",
                title="Image Loading Failed"
            )

    return ""
