from visualization.experiment import experiment
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from base64 import b64encode
import io

import pandas as pd

app = Dash(__name__)

buffer = io.StringIO()

# results = pd.read_csv('/home/isabella/.bittensor/bittensor/visualization/results.csv')
# results = results.sort_values(by =['time'])
# results = results.reset_index()
# print(results)
# grouped_result = results.groupby(['batch_size', 'sequence_len']).mean()

# # plots
# respond_time = px.line(results, y='time', title="Sorted respond time")
# respond_code = px.histogram(results, x = 'code')

app.layout = html.Div(children=[
    html.H1(children='Network vis'),

    html.Div(children='~~~ Respond time check ~~~'),

    dcc.Graph(id="respond-time"),
    dcc.Checklist(
        id="respond-time-filter",
        options=[10, 110],
        value=[10, 110],
        inline=True
    ),
    
    # html.P("Mean:"),
    # dcc.Slider(id="mean", min=-3, max=3, value=0, marks={-3: '-3', 3: '3'}),
    # html.P("Standard Deviation:"),
    # dcc.Slider(id="std", min=1, max=3, value=1, marks={1: '1', 3: '3'}),

])

@app.callback(Output("respond-time", "figure"), Input("respond-time-filter", "value"))
def respond_time_plot(sequence_lens):
    results = pd.read_csv('/home/isabella/.bittensor/bittensor/visualization/results.csv')
    results = results.sort_values(by =['time'])
    results = results.reset_index(drop = True)
    mask = results.sequence_len.isin(sequence_lens)
    results_masked = results[mask].reset_index()
    # results_masked['index'] = results_masked['index'] % 2000

    print(results_masked)
    fig = px.line(results_masked, x = 'index', y="time", color='sequence_len', title="Sorted respond time")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
