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
            ui.column(4, ui.h4("Immediate Memory: List Learning Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("imi_llts", "", "", min=1, max=40), style="width: 100px;"),
            ui.column(6, ui.output_text_verbatim("validation_imi_llts")),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Immediate Memory: Story Memory Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("imi_smts", "", "", min=1, max=24), style="width: 100px;"),
            ui.column(6, ui.output_text_verbatim("validation_imi_smts")),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Language: Semantic Fluency Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("lis_sf", "", "", min=1, max=40), style="width: 100px;"),
            ui.column(6, ui.output_text_verbatim("validation_lis_sf")),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Delayed Memory: List Recall Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("dmi_lrts", "", "", min=1, max=10), style="width: 100px;"),
            ui.column(6, ui.output_text_verbatim("validation_dmi_lrts")),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Delayed Memory: List Recognition Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("rt_score", "", "", min=1, max=20), style="width: 100px;"),
            ui.column(6, ui.output_text_verbatim("validation_rt_score")),
            style="margin-right: 105px;"
        ),
        ui.row(
            ui.column(4, ui.h4("Delayed Memory: Story Recall Score", style="font-size: 16px;")),
            ui.column(4, ui.input_numeric("dmi_srts", "", "", min=1, max=12), style="width: 100px;"),
            ui.column(6, ui.output_text_verbatim("validation_dmi_srts")),
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
        scores = [0, input.imi_llts(), input.imi_smts()] + [0] * 5 + \
            [input.lis_sf()] + [0] * 4 + \
            [input.dmi_lrts(), input.rt_score(), input.dmi_srts()] + [0] * 7

        collabs = ['imi_score', 'imi_llts', 'imi_smts', 'vci_score',
                   'vci_fcts', 'vci_lots', 'lis_score', 'lis_pic', 'lis_sf',
                   'ai_score', 'ai_ds', 'ai_coding', 'dmi_score', 'dmi_lrts',
                   'rt_score', 'dmi_srts', 'dmi_frts', 'tsi_score',
                   'list_hits', 'list_fp', 'age', 'sex_id', 'education']

        return pd.DataFrame([scores], columns=collabs)

    # *** Return the probabilistic scores per class
    @reactive.Calc
    def probability_scores():
        scores = rbans_scores()
        proba = predict_proba(scores)
        return proba

    # ** Add a function to test if all input values are valid
    @reactive.Calc
    def check_input():
        return (
            (input.imi_llts() is not None and 0 <= input.imi_llts() <= 40) and
            (input.imi_smts() is not None and 0 <= input.imi_smts() <= 24) and
            (input.lis_sf() is not None and 0 <= input.lis_sf() <= 40) and
            (input.dmi_lrts() is not None and 0 <= input.dmi_lrts() <= 10) and
            (input.rt_score() is not None and 0 <= input.rt_score() <= 20) and
            (input.dmi_srts() is not None and 0 <= input.dmi_srts() <= 12)
        )


    # ** Predict the binary class label
    @output()
    @render.text
    def binary_classification():

        if check_input():

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

        if check_input():

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

        if check_input():

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
                    legend_traceorder="normal",
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(
                        range=[-0.01, 1.01],
                        tick0=0,
                        dtick=0.1,
                        ticks="outside",
                        ticklen=6,
                        tickwidth=2,
                        tickcolor='black'
                    )
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

        The model presented here was trained on 95 subjects (37 cognitively unimpaired subjects, 31 subjects with mild cognitive impairment, and 27 subjects with Alzheimer's disease). All subjects were between 65 and 91 years of age, 56.6% of the sample was female, subjects completed between 12 and 20 years of education, 97.8% of the sample was White/Caucasian and 98.5% of the sample were not Latino or Hispanic. The normative data used to calculate the Index and Total Scale scores for the subjects in the sample come from the original 1998 RBANS manual, which only corrected for age. The model is not intended to replace current diagnostic tools, but it could provide some consistency of diagnostic decisions in clinical settings and research projects.<br><br>

        Information on the data used to train this model, how the model was trained, and how to interpret the outcomes, as well as a discussion of the model performance in terms of classification accuracy following cross-validation is discussed in the manuscript: <i>Classification of Mild Cognitive Impairment and Alzheimer’s Disease Using the Repeatable Battery for the Assessment of Neuropsychological Status</i> by <u>Vincent Koppelmans</u>, <u>Tolga Tasdizen</u>, and <u>Kevin Duff</u> (currently submitted for publication; upon publication a link to the article will be included here)."""

        return html_text

    @output()
    @render.text
    def validation_imi_llts():
        value = input.imi_llts()
        if value is None:
            return "(0-40)"
        elif not (0 <= value <= 40):
            return "INCORRECT: Value must be between 0 and 40!"
        else:
            return "Value is valid."

    @output()
    @render.text
    def validation_imi_smts():
        value = input.imi_smts()
        if value is None:
            return "(0-24)"
        elif not (0 <= value <= 24):
            return "INCORRECT: Value must be between 0 and 24!"
        else:
            return "Value is valid."

    @output()
    @render.text
    def validation_lis_sf():
        value = input.lis_sf()
        if value is None:
            return "(0-40)"
        elif not (0 <= value <= 40):
            return "INCORRECT: Value must be between 0 and 40!"
        else:
            return "Value is valid."

    @output()
    @render.text
    def validation_dmi_lrts():
        value = input.dmi_lrts()
        if value is None:
            return "(0-10)"
        elif not (0 <= value <= 10):
            return "INCORRECT: Value must be between 0 and 10!"
        else:
            return "Value is valid."

    @output()
    @render.text
    def validation_rt_score():
        value = input.rt_score()
        if value is None:
            return "(0-20)"
        elif not (0 <= value <= 20):
            return "INCORRECT: Value must be between 0 and 20!"
        else:
            return "Value is valid."

    @output()
    @render.text
    def validation_dmi_srts():
        value = input.dmi_srts()
        if value is None:
            return "(0-12)"
        elif not (0 <= value <= 12):
            return "INCORRECT: Value must be between 0 and 12!"
        else:
            return "Value is valid."


# * Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(show_error=False)
