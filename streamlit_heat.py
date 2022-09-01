import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

st.title("Heatmap")

with st.expander("See source code"):
    with st.echo():
        filepath = r"F:\NY-city-siniestralidad\crashes.csv"
        m = leafmap.Map(center=[40.730610, -73.935242], zoom=9, tiles="stamentoner")
        #m = leafmap.Map(center=[40.730610, -73.935242], zoom=9, tiles="stamentoner")
        m.add_heatmap(
            filepath,
            latitude="latitude",
            longitude="longitude",
            value="check",
            name="Heat map",
            radius=14,
        )
m.to_streamlit(height=700)
