import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_data():
    df =  pd.read_csv(r'F:\NY-city-siniestralidad\ny_city_data.csv')[['longitude','latitude']].dropna()
    df['latitude'] = round(df['latitude'], 2)
    df['longitude'] = round(df['longitude'], 2)
    return df

# def load_data():
#     return pd.DataFrame(
#         np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
#     )

df = load_data()

st.subheader('Siniestralidad Vial')

# Map to show the physical locations of Crime for the selected day.
midpoint = (np.average(df["latitude"]), np.average(df["longitude"]))

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=40.730610,
            longitude=-73.935242,
            zoom=9,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position="[longitude, latitude]",
                radius=200,
                elevation_scale=6,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[longitude, latitude]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)