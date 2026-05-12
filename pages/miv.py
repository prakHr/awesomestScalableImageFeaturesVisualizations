from dash import (
    html,
    Input,
    Output,
    State,
    callback
)

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc

import neurova as nv

import base64
import matplotlib.pyplot as plt
import io as pyio

from neurova import (
    io,
    transform,
    core,
    datasets
)

import numpy as np
import os


dash.register_page(
    __name__,
    path="/miv",
    name="MIV"
)


SUPPORTED_EXTENSIONS_MIV = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif"
)


def get_image_files_miv(folder_path_miv):

    return [

        os.path.join(
            folder_path_miv,
            file_miv
        )

        for file_miv in os.listdir(
            folder_path_miv
        )

        if file_miv.lower().endswith(
            SUPPORTED_EXTENSIONS_MIV
        )
    ]


def convert_array_to_base64_miv(arr_miv):

    fig_miv, ax_miv = plt.subplots()

    ax_miv.imshow(arr_miv)
    ax_miv.axis("off")

    buf_miv = pyio.BytesIO()

    plt.savefig(
        buf_miv,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    buf_miv.seek(0)

    img_base64_miv = base64.b64encode(
        buf_miv.read()
    ).decode("utf-8")

    plt.close(fig_miv)

    return (
        f"data:image/png;base64,"
        f"{img_base64_miv}"
    )


def create_grid_data_miv(
    image_files_miv,
    operation_miv,
    width_miv=None,
    height_miv=None,
    angle_miv=None
):

    row_data_miv = []

    for file_miv in image_files_miv:

        rgb_image_miv = io.imread(
            file_miv
        )

        filename_miv = os.path.basename(
            file_miv
        )

        if operation_miv == "grayscale":

            arr_miv = core.to_grayscale(
                rgb_image_miv
            )

            row_data_miv.append({

                "filename": filename_miv,

                "operation": "Grayscale",

                "image": convert_array_to_base64_miv(
                    arr_miv
                )
            })

        elif operation_miv == "hsv":

            arr_miv = core.convert_color_space(

                rgb_image_miv,

                core.ColorSpace.HSV,

                from_space=core.ColorSpace.RGB
            )

            row_data_miv.append({

                "filename": filename_miv,

                "operation": "HSV",

                "image": convert_array_to_base64_miv(
                    arr_miv
                )
            })

        elif operation_miv == "rgb":

            row_data_miv.append({

                "filename": filename_miv,

                "operation": "RGB",

                "image": convert_array_to_base64_miv(
                    rgb_image_miv
                )
            })

        elif operation_miv == "resize":

            arr_miv = transform.resize(

                rgb_image_miv,

                (
                    int(width_miv),
                    int(height_miv)
                )
            )

            row_data_miv.append({

                "filename": filename_miv,

                "operation": (
                    f"Resize "
                    f"{width_miv}x{height_miv}"
                ),

                "image": convert_array_to_base64_miv(
                    arr_miv
                )
            })

        elif operation_miv == "rotate":

            arr_miv = transform.rotate(

                rgb_image_miv,

                int(angle_miv)
            )

            row_data_miv.append({

                "filename": filename_miv,

                "operation": (
                    f"Rotate "
                    f"{angle_miv}°"
                ),

                "image": convert_array_to_base64_miv(
                    arr_miv
                )
            })

        elif operation_miv == "flip":

            flipped_h_miv = np.flip(
                rgb_image_miv,
                axis=1
            )

            flipped_v_miv = np.flip(
                rgb_image_miv,
                axis=0
            )

            flipped_both_miv = np.flip(
                flipped_v_miv,
                axis=1
            )

            row_data_miv.extend([

                {

                    "filename": filename_miv,

                    "operation": (
                        "Horizontal Flip"
                    ),

                    "image": convert_array_to_base64_miv(
                        flipped_h_miv
                    )
                },

                {

                    "filename": filename_miv,

                    "operation": (
                        "Vertical Flip"
                    ),

                    "image": convert_array_to_base64_miv(
                        flipped_v_miv
                    )
                },

                {

                    "filename": filename_miv,

                    "operation": (
                        "Both Axes Flip"
                    ),

                    "image": convert_array_to_base64_miv(
                        flipped_both_miv
                    )
                }
            ])

    return row_data_miv


column_defs_miv = [

    {

        "headerName": "Filename",

        "field": "filename",

        "sortable": True,

        "filter": True,

        "resizable": True,

        "flex": 1
    },

    {

        "headerName": "Operation",

        "field": "operation",

        "sortable": True,

        "filter": True,

        "resizable": True,

        "flex": 1
    },

    {

        "headerName": "Image",

        "field": "image",

        "cellRenderer": "ImageRenderer",

        "autoHeight": True,

        "flex": 2
    }
]


layout = dmc.MantineProvider(

    dmc.Container(

        [

            dmc.Title(
                "Dash AG Grid Image Viewer",
                order=2,
                mb="md"
            ),

            dmc.TextInput(
                id="input-folder-path-miv",
                placeholder="Enter folder path",
                label="Folder Path",
                w=500
            ),

            dmc.Space(h=20),

            dmc.Group(

                [

                    dmc.Button(
                        "Show Grayscale Images",
                        id="submit-button-miv",
                        n_clicks=0
                    ),

                    dmc.Button(
                        "Show HSV Images",
                        id="submit-button2-miv",
                        n_clicks=0
                    ),

                    dmc.Button(
                        "Show RGB Images",
                        id="submit-button3-miv",
                        n_clicks=0
                    ),

                    dmc.Button(
                        "Show Flipped Images",
                        id="submit-button6-miv",
                        n_clicks=0
                    )

                ]

            ),

            dmc.Space(h=20),

            dmc.Group(

                [

                    dmc.TextInput(
                        id="input-resize-h-text-miv",
                        placeholder="Enter height",
                        label="Resize Height",
                        w=200
                    ),

                    dmc.TextInput(
                        id="input-resize-w-text-miv",
                        placeholder="Enter width",
                        label="Resize Width",
                        w=200
                    ),

                    dmc.Button(
                        "Show Resized Images",
                        id="submit-button4-miv",
                        n_clicks=0,
                        mt=25
                    )

                ]

            ),

            dmc.Space(h=20),

            dmc.Group(

                [

                    dmc.TextInput(
                        id="input-rotation-text-miv",
                        placeholder="Enter rotation angle",
                        label="Rotation Angle",
                        w=200
                    ),

                    dmc.Button(
                        "Show Rotated Images",
                        id="submit-button5-miv",
                        n_clicks=0,
                        mt=25
                    )

                ]

            ),

            dmc.Space(h=30),

            dag.AgGrid(

                id="image-grid-miv",

                columnDefs=column_defs_miv,

                rowData=[],

                defaultColDef={

                    "wrapText": True
                },

                dashGridOptions={

                    "rowHeight": 300,

                    "pagination": True,

                    "paginationPageSize": 4
                },

                style={

                    "height": "900px",
                    "width": "100%"
                },

                className="ag-theme-alpine",

                dangerously_allow_code=True
            )

        ],

        size="xl",
        pt=40

    )

)


@callback(
    Output(
        "image-grid-miv",
        "rowData"
    ),

    Input(
        "submit-button-miv",
        "n_clicks"
    ),

    Input(
        "submit-button2-miv",
        "n_clicks"
    ),

    Input(
        "submit-button3-miv",
        "n_clicks"
    ),

    Input(
        "submit-button4-miv",
        "n_clicks"
    ),

    Input(
        "submit-button5-miv",
        "n_clicks"
    ),

    Input(
        "submit-button6-miv",
        "n_clicks"
    ),

    State(
        "input-folder-path-miv",
        "value"
    ),

    State(
        "input-resize-w-text-miv",
        "value"
    ),

    State(
        "input-resize-h-text-miv",
        "value"
    ),

    State(
        "input-rotation-text-miv",
        "value"
    ),

    prevent_initial_call=True
)
def update_grid_miv(

    grayscale_clicks_miv,

    hsv_clicks_miv,

    rgb_clicks_miv,

    resize_clicks_miv,

    rotate_clicks_miv,

    flip_clicks_miv,

    folder_path_miv,

    width_miv,

    height_miv,

    angle_miv
):

    ctx_miv = dash.callback_context

    if not ctx_miv.triggered:

        return []

    button_id_miv = (
        ctx_miv.triggered[0]["prop_id"]
        .split(".")[0]
    )

    if not folder_path_miv:

        return []

    try:

        image_files_miv = get_image_files_miv(
            folder_path_miv
        )

        if button_id_miv == "submit-button-miv":

            return create_grid_data_miv(

                image_files_miv,

                "grayscale"
            )

        elif button_id_miv == "submit-button2-miv":

            return create_grid_data_miv(

                image_files_miv,

                "hsv"
            )

        elif button_id_miv == "submit-button3-miv":

            return create_grid_data_miv(

                image_files_miv,

                "rgb"
            )

        elif button_id_miv == "submit-button4-miv":

            return create_grid_data_miv(

                image_files_miv,

                "resize",

                width_miv=width_miv,

                height_miv=height_miv
            )

        elif button_id_miv == "submit-button5-miv":

            return create_grid_data_miv(

                image_files_miv,

                "rotate",

                angle_miv=angle_miv
            )

        elif button_id_miv == "submit-button6-miv":

            return create_grid_data_miv(

                image_files_miv,

                "flip"
            )

    except Exception as e_miv:

        return [

            {

                "filename": "Error",

                "operation": "Exception",

                "image": str(e_miv)
            }

        ]

    return []


dash.clientside_callback(

    """
    function(rowData) {

        window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

        window.dashAgGridComponentFunctions.ImageRenderer = function(props) {

            return React.createElement(

                'img',

                {

                    src: props.value,

                    style: {

                        width: '250px',

                        borderRadius: '10px',

                        padding: '5px'
                    }
                }
            );
        };

        return window.dash_clientside.no_update;
    }
    """,

    Output(
        "image-grid-miv",
        "id"
    ),

    Input(
        "image-grid-miv",
        "rowData"
    )
)