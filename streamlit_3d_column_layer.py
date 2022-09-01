import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_data():
    df =  pd.read_csv('victimas.csv')
    df = df[(df['crash_date'] > "2020-07-01")]

    df['total_victims'] = df['total_victims'].astype(int)
    df = df[['crash_date',
             'latitude',
             'longitude',
             'total_victims',
             'total_killed',
             'total_injured']].dropna()
    #df['latitude'] = round(df['latitude'], 2)
    #df['longitude'] = round(df['longitude'], 2)
    return df

df = load_data()

st.subheader('Siniestralidad Vial')

#FECHASS
def between_date(data,start,end):
    mask = (data['crash_date'] >= start) & (data['crash_date'] <= end)
    return data.loc[mask]

dates = pd.to_datetime(df['crash_date'])
date_selection = st.sidebar.slider('Fecha:',
                            min_value= dates.max().date(),  
                            max_value= dates.max().date(),
                            value=(dates.min().date(),dates.max().date()))
df = between_date(df,str(date_selection[0]),str(date_selection[1]))

#FEATURES
st.subheader("Feature")
options = df.drop(columns=(['latitude','longitude','crash_date'])).columns.to_list()
map_selection = st.selectbox(label= 'Seleccionar datos de:',options=options)

if map_selection == 'total_killed':
    elevation_scale = 100
    min_value = float(0)
else:
    elevation_scale = 20
    min_value = float(3)

#GRAVEDAD
gravedad = st.sidebar.slider('Gravedad:',
                            min_value = min_value,
                            max_value = float(df[map_selection].max()),
                            value=(min_value,float(df[map_selection].max())))
df = df[(df[map_selection] >= gravedad[0]) & (df[map_selection] <= gravedad[1])]

column_layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position=["longitude", "latitude"],
    get_elevation=f"{map_selection}*10",
    elevation_scale=elevation_scale,
    radius=300,
    get_fill_color=[f"{map_selection} * 30", f"{map_selection} * 10", f"{map_selection} * 5"],
    pickable=True,
    auto_highlight=True,
)

tooltip = {
     "html": "Location: {longitude}</br> Latitude: {latitude} </br> Count: {total_victims}</br> Feature: {total_victims}",
     "style": {"background": "steelblue", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
}

r = pdk.Deck(
    column_layer,
    map_style=None,
    initial_view_state=pdk.ViewState(
            latitude=40.730610,
            longitude=-73.935242,
            zoom=10,
            pitch=30,
        ),
    tooltip=tooltip,
)

st.pydeck_chart(r)

