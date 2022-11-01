import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sqlalchemy import create_engine


coon = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)
FIELD = pd.read_sql('SELECT * FROM FIELDcd1515101', coon)


# FIELD = FIELD[FIELD['x'] == FIELD['x'].unique()[52]]
# fig = go.Figure(data=go.Contour(
#     z=FIELD['FIELDRe'].values.reshape([101, 101])
# ))

fig = go.Figure(data=go.Volume(
     x=FIELD['x'],
     y=FIELD['y'],
     z=FIELD['z'],
     value=FIELD['FIELDRe'],
     isomin=0.1,
     isomax=0.8,
     opacity=0.1))
fig.show()


