import dill
from shiny import ui, App, render, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# * Import the trained model
with open('rbans_model.dill.pkl', 'rb') as file:
    predict, predict_proba = dill.load(file)

# * Read RBANS data from user input
# Define the UI
app_ui = ui.page_fluid(
    # ** Title
    ui.panel_title("RBANS Classification Tool"),

    # ** Read input data from user
    ui.TagList(
        ui.row(
            ui.column(4, ui.h4("Immediate Memory INDEX Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("imi_score", "", "", min=1, max=160), style="width: 100px;"),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("List Learning Total Raw Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("imi_llts", "", "", min=1, max=40), style="width: 100px;"),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Language INDEX Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("lis_score", "", "", min=1, max=160), style="width: 100px;"),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Semantic Fluency Total Raw Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("lis_sf", "", "", min=1, max=160), style="width: 100px;"),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Delayed Memory INDEX Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("dmi_score", "", "", min=1, max=160), style="width: 100px;"),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Story Recall Total Raw Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("dmi_srts", "", "", min=1, max=12), style="width: 100px;"),
            style="margin-right: 105px;"
        )
    ),

    # ** Define which objects to expect
    ui.tags.hr(),
    ui.HTML("<b>Predicted Label</b>"),
    ui.output_text("binary_classification"),
    ui.tags.hr(),
    ui.HTML("<b>Class Specific Probabilities</b>"),
    ui.output_ui("probabilistic_classification"), # _ui instead of _text for html rendering
    output_widget("bar_plot"),
    ui.tags.hr(),
    ui.output_ui("reference")
)

# * Define the server logic
def server(input, output, session):

    # ** Add a reactive object for returning calculated calues
    # *** Return a data frame with the entered rbans scores in a larger DF
    @reactive.Calc
    def rbans_scores():
        scores = [
            input.imi_score(), input.imi_llts()] + [0] * 4 + \
            [input.lis_score(), 0, input.lis_sf(), 0, 0, 0, input.dmi_score(),
             0, 0, input.dmi_srts()] + [0] * 7

        collabs = ['imi_score', 'imi_llts', 'imi_smts', 'vci_score',
                   'vci_fcts', 'vci_lots', 'lis_score', 'lis_pic', 'lis_sf',
                   'ai_score', 'ai_ds', 'ai_coding', 'dmi_score', 'dmi_lrts',
                   'rt_score', 'dmi_srts', 'dmi_frts', 'tsi_score',
                   'list_hits', 'list_fp', 'age', 'sex_id', 'education']

        return pd.DataFrame([scores], columns=collabs)

    # *** Returnt the probabilistic scores per class
    def probability_scores():
        scores = rbans_scores()
        proba = predict_proba(scores)

        return proba

    # ** Predict the binary class label
    @output()
    @render.text
    def binary_classification():

        # *** Pull rbans scores
        scores = rbans_scores()

        try:
            mypredicted = {
                0:"Cognitively Unimpaired",
                1:"Mild Cognitive Impairment",
                2:"Alzheimer's Disease"}[
                    predict(scores)[0]
                ]

            return f"{mypredicted}"

        except Exception:
            return ""

    # ** Predict the probablistic class labels
    @output()
    @render.text
    def probabilistic_classification():

        try:
            proba = probability_scores()
            return (
                f"<br>"
                f"Cognitively Unimpaired:    {round(proba[0][0], 3)}<br>"
                f"Mild Cognitive Impairment: {round(proba[0][1], 3)}<br>"
                f"Alzheimer's Disease:       {round(proba[0][2], 3)}<br>"
            )

        except Exception:
            return ""

    # ** Display the probability values as a bar graph
    @output()
    @render_widget
    def bar_plot():

        try:
            # *** Grab the data
            proba = probability_scores()

            # *** Align values, labels, and colors
            values = [proba[0, 0], proba[0, 1], proba[0, 2]]
            labels = ['CU', 'MCI', 'AD']
            colors = ['#FFFFFF', '#E4937B', '#890000']

            # ** Create the figure with a single bar with segmented colors
            fig = go.Figure()

            # ** Add segments of the bar
            for i in range(len(values)):
                fig.add_trace(go.Bar(
                    y=[0],           # Since it's a single bar for the y-axis
                    x=[values[i]],   # Width of each segment
                    name=labels[i],  # Label for each segment
                    orientation='h',
                    marker=dict(color=colors[i]),
                    hoverinfo='name+x'  # Show name and value on hover
                ))

            # ** Update figure layout
            fig.update_layout(
                title='',
                xaxis_title='',
                yaxis=dict(showticklabels=False),  # Hide y-ticks
                barmode='stack',                   # Stack the bars
                width=320,                         # Set width of the figure
                height=160,                        # Set height of the figure
                legend=dict(
                    orientation="h",  # Horizontal layout
                    yanchor="bottom",  # Anchor at the bottom
                    y=-0.8,  # Adjust Y position (negative to move below the plot)
                    xanchor="center",  # Center the legend on the X axis
                    x=0.5  # Center the legend based on the entire figure width
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(range=[-0.01, 1.01]),
            )
            return fig

        except Exception:
            # In this case, return an empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis=dict(
                    showticklabels=False, showline=False, zeroline=False,
                    gridcolor='rgba(255, 255, 255, 0)'
                ),
                yaxis=dict(
                    showticklabels=False, showline=False, zeroline=False,
                    gridcolor='rgba(255, 255, 255, 0)'
                ),
                plot_bgcolor='rgba(255, 255, 255, 0)',  # Transparent backgrnd
                width=10,
                height=10,
                margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
            )
            return empty_fig

    # ** Predict the probablistic class labels
    @output()
    @render.text
    def reference():
        html_text = f"""<b>Background</b><br><a href="https://github.com/vnckppl/APPE_RBANS_Classification">THIS</a> github repository contains the classification model for mild cognitive impairment and Alzheimer’s disease based on performance on the <i>Repeatable Battery for the Assessment of Neuropsychological Status</i> (RBANS) presented here.<br><br>

        Information on the data used to train this model, how the model was trained, and how to interpret the outcomes, as well as a discussion of the model performance in terms of classification accuracy following cross-validation is discussed in the manuscript: <i>Classification of Mild Cognitive Impairment and Alzheimer’s Disease Using the Repeatable Battery for the Assessment of Neuropsychological Status</i> by <u>Vincent Koppelmans</u>, <u>Tolga Tasdizen</u>, and <u>Kevin Duff</u> (currently submitted for publication; upon publication a link to the article will be included here)."""

        return html_text


# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(show_error=False)
