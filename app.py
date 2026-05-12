from dash import Dash, html, dcc, Input, Output
import dash_mantine_components as dmc
import dash
import multiprocessing

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True
)

app.layout = dmc.MantineProvider(

    dmc.Container(

        [

            # URL tracker
            dcc.Location(id="url", refresh=True),

            dmc.Title(
                "Neurova Image Processing App",
                order=1,
                mb="lg"
            ),

            dmc.Group(

                [
                    dmc.Anchor("single_image_visualizations", href="/siv"),
                    dmc.Anchor("generate_single_image_features", href="/gif"),
                    dmc.Anchor("multiple_image_visualizations", href="/miv"),
                    dmc.Anchor("generate_multiple_image_features", href="/gmf"),
                ],

                mb="xl"
            ),

            dash.page_container

        ],

        pt=40
    )
)


# Redirect root URL "/" to "/siv"
@app.callback(
    Output("url", "pathname"),
    Input("url", "pathname"),
    prevent_initial_call=False
)
def redirect_home(pathname):
    if pathname == "/":
        return "/siv"
    return pathname


if __name__ == "__main__":

    multiprocessing.freeze_support()

    app.run(
        debug=True,
        use_reloader=False
    )