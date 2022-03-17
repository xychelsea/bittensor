import plotly.express as px
from visualization.client import experiment


fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
fig.write_html('/root/.bittensor/bittensor/visualization/vis.html', auto_open=True)