import plotly.express as px
from visualization.experiment import experiment


from dash import Dash, dcc, html
import plotly.express as px
from base64 import b64encode
import io

import pandas as pd

app = Dash(__name__)

buffer = io.StringIO()

# results = experiment().run()
results = pd.read_csv('/home/isabella/.bittensor/bittensor/visualization/results.csv')
results = results.sort_values(by =['time'])
grouped_result = results.groupby(['batch_size', 'sequence_len']).mean()

respond_time = px.line(results, y='time')
respond_code = px.histogram(results, x = 'code')

app.layout = html.Div(children=[
    html.H1(children='Another Bittensor exploere'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    html.H3('Respond time'),
    dcc.Graph(id="respond-time", figure = respond_time),
    html.P("Mean:"),
    dcc.Slider(id="mean", min=-3, max=3, value=0, marks={-3: '-3', 3: '3'}),
    html.P("Standard Deviation:"),
    dcc.Slider(id="std", min=1, max=3, value=1, marks={1: '1', 3: '3'}),


])

if __name__ == '__main__':
    app.run_server(debug=True)
