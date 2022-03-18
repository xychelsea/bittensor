from visualization.experiment import experiment
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from base64 import b64encode
import io
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import os

app = Dash(__name__)

buffer = io.StringIO()

result_path = os.path.expanduser('~/.bittensor/bittensor/visualization/exp_results/')

result = pd.read_csv(os.path.join(result_path, 'results.csv'))
setup = result.drop(['uid', 'code', 'time', 'block'], axis = 1).drop_duplicates()
print(setup)

app.layout = html.Div(children=[
    html.H1(children='Network vis'),

    # dcc.Graph(id="respond-time"),
    dcc.Graph(id="dummy-respond-time-line"),
    dcc.Graph(id="dummy-respond-time-hist"),
    dcc.Graph(id="dummy-respond-time-heatmap"),
    dcc.Graph(id="dummy-respond-time-cat"),
    dcc.Checklist(
        id="respond-time-filter",
        options=[10, 110],
        value=[10, 110],
        inline=True
    ),
    
])

@app.callback(Output("dummy-respond-time-line", "figure"), Input("respond-time-filter", "value"))
def dummy_respond_time_line(sequence_lens):
    fig = make_subplots(rows=1, cols=2)
    result = pd.read_json(os.path.join(result_path, 'mock_result.txt'))
    result = result.transpose()
    result = result.sort_values(by =['time'])
    result = result.reset_index(drop = True)
    result = result.astype({'code': 'str'})
    fig = px.line(result, y="time", title="Respond time")
    fig.update_layout( width = 1500, height = 1000)
    
    return fig

@app.callback(Output("dummy-respond-time-hist", "figure"), Input("respond-time-filter", "value"))
def dummy_respond_time_hist(sequence_lens):
    result = pd.read_json(os.path.join(result_path, 'mock_result.txt'))
    result = result.transpose()
    result = result.astype({'code': 'str'})
    fig = px.histogram(result, x="code", title="Respond code")
    fig.update_layout( width = 1000, height = 1000)
    return fig

@app.callback(Output("dummy-respond-time-heatmap", "figure"), Input("respond-time-filter", "value"))
def dummy_respond_time_heatmap(sequence_lens):
    result = pd.read_json(os.path.join(result_path, 'mock_result.txt'))
    result = result.transpose()
    times = result['time'].to_numpy()
    times = times.reshape((40,50))
    times = pd.DataFrame(times)
    times = 12- times
    fig = px.imshow(times, color_continuous_scale='Greens', title="Respond time")
    
    fig.update_layout( width = 4000, height = 4000)
    return fig

@app.callback(Output("dummy-respond-time-cat", "figure"), Input("respond-time-filter", "value"))
def dummy_respond_time_heatmap(sequence_lens):
    result = pd.read_csv(os.path.join(result_path, 'result.csv'))
    result = result[result['batch_size'] == 3]

    result = result.sort_values(by =['batch_size', 'sequence_len', 'time'])
    result['rank'] = [ r % 2000 for r in range(0, len(result) )]


    # result = result.set_index(['batch_size', 'sequence_len'])
    # result = result.sort_values(by =['time'])
    # result['rank'] = [ r % 2000 for r in range(0, len(result) )]
    # result = result.reset_index()
    # print(result)
    # inspect = result[result['sequence_len'] == 80]
    # inspect.to_csv('/home/isabella/.bittensor/bittensor/visualization/exp_results/inspect.csv')
    # print(inspect)

    fig = px.line(result, x = 'rank', y='time', color = 'sequence_len' ,title="Respond time, batch_size = 3, with varying sequence_len")
    fig.update_layout( width = 1500, height = 1000)
    return fig

if __name__ == '__main__':
    app.run_server(host = '143.198.236.213', debug=True)
    # app.run_server( debug=True)
