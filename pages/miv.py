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

from neurova import (
    io,
    transform,
    core
)

import pandas as pd
import numpy as np

import multiprocessing as mp
from multiprocessing import Pool

import matplotlib.pyplot as plt
import base64
import io as pyio
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


def process_image_miv(args_miv):

    row_miv = args_miv[0]

    operation_miv = args_miv[1]

    width_miv = args_miv[2]

    height_miv = args_miv[3]

    angle_miv = args_miv[4]

    file_miv = row_miv["filepath"]

    filename_miv = row_miv["filename"]

    rgb_image_miv = io.imread(
        file_miv
    )

    output_rows_miv = []

    if operation_miv == "grayscale":

        arr_miv = core.to_grayscale(
            rgb_image_miv
        )

        output_rows_miv.append({

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

        output_rows_miv.append({

            "filename": filename_miv,

            "operation": "HSV",

            "image": convert_array_to_base64_miv(
                arr_miv
            )
        })

    elif operation_miv == "rgb":

        output_rows_miv.append({

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

        output_rows_miv.append({

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

        output_rows_miv.append({

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

        output_rows_miv.extend([

            {

                "filename": filename_miv,

                "operation": "Horizontal Flip",

                "image": convert_array_to_base64_miv(
                    flipped_h_miv
                )
            },

            {

                "filename": filename_miv,

                "operation": "Vertical Flip",

                "image": convert_array_to_base64_miv(
                    flipped_v_miv
                )
            },

            {

                "filename": filename_miv,

                "operation": "Both Axes Flip",

                "image": convert_array_to_base64_miv(
                    flipped_both_miv
                )
            }
        ])

    return output_rows_miv


def create_grid_data_miv(
    image_files_miv,
    operation_miv,
    width_miv=None,
    height_miv=None,
    angle_miv=None
):

    df_miv = pd.DataFrame({

        "filepath": image_files_miv,

        "filename": [

            os.path.basename(
                file_miv
            )

            for file_miv in image_files_miv
        ]
    })

    args_list_miv = [

        (
            row_miv,
            operation_miv,
            width_miv,
            height_miv,
            angle_miv
        )

        for _, row_miv in df_miv.iterrows()
    ]

    num_cores_miv = max(
        1,
        mp.cpu_count() - 1
    )

    with Pool(num_cores_miv) as pool_miv:

        results_miv = pool_miv.map(

            process_image_miv,

            args_list_miv
        )

    flattened_results_miv = [

        item_miv

        for sublist_miv in results_miv

        for item_miv in sublist_miv
    ]

    return flattened_results_miv


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
                "Multiprocessing Image Viewer",
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
                        id="submit-button-miv"
                    ),

                    dmc.Button(
                        "Show HSV Images",
                        id="submit-button2-miv"
                    ),

                    dmc.Button(
                        "Show RGB Images",
                        id="submit-button3-miv"
                    ),

                    dmc.Button(
                        "Show Flipped Images",
                        id="submit-button6-miv"
                    )
                ]
            ),

            dmc.Space(h=30),

            dag.AgGrid(

                id="image-grid-miv",

                columnDefs=column_defs_miv,

                rowData=[],

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
        "submit-button6-miv",
        "n_clicks"
    ),

    State(
        "input-folder-path-miv",
        "value"
    ),

    prevent_initial_call=True
)
def update_grid_miv(

    grayscale_clicks_miv,

    hsv_clicks_miv,

    rgb_clicks_miv,

    flip_clicks_miv,

    folder_path_miv
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

        operation_map_miv = {

            "submit-button-miv": "grayscale",

            "submit-button2-miv": "hsv",

            "submit-button3-miv": "rgb",

            "submit-button6-miv": "flip"
        }

        operation_miv = operation_map_miv.get(
            button_id_miv
        )

        return create_grid_data_miv(

            image_files_miv,

            operation_miv
        )

    except Exception as e_miv:

        return [

            {

                "filename": "Error",

                "operation": "Exception",

                "image": str(e_miv)
            }
        ]


dash.clientside_callback(

    """
    function(rowData) {

        window.dashAgGridComponentFunctions =
        window.dashAgGridComponentFunctions || {};

        window.dashAgGridComponentFunctions.ImageRenderer =
        function(props) {

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
