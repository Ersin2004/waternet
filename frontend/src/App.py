import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pyproj import Transformer
import altair as alt
import glob

st.set_page_config(page_title="Waternet Dashboard", layout="wide")


import os, glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

files_chem = glob.glob(os.path.join(project_root, "data", "out", "chem", "*.csv"))
# files_bio = glob.glob(os.path.join(project_root, "data", "out", "bio", "*.csv"))

files = files_chem 
# + files_bio
print("Aantal bestanden gevonden:", len(files))


df_list = [pd.read_csv(f, sep=";", encoding="latin1") for f in files]
df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.strip()


parameters = df["fewsparameternaam"].dropna().unique()
selected_param = st.selectbox("Kies een parameter:", parameters)

df_filtered_param = df[df["fewsparameternaam"] == selected_param]


df_filtered_param["year"] = pd.to_datetime(df_filtered_param["datum"]).dt.year
years = df_filtered_param["year"].unique()
selected_year = st.select_slider("Kies een jaar:", options=sorted(years))
df_filtered_year = df_filtered_param[df_filtered_param["year"] == selected_year]


transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
df_filtered_year["lon"], df_filtered_year["lat"] = transformer.transform(
    df_filtered_year["x_location"].values, df_filtered_year["y_location"].values
)


col1, col2 = st.columns([2, 1])


with col1:
    m = folium.Map(location=[52.37, 4.90], zoom_start=12, tiles="CartoDB positron")

    heat_data = [
        [row["lat"], row["lon"], row["meetwaarde"]]
        for _, row in df_filtered_year.iterrows()
        if pd.notnull(row["lat"]) and pd.notnull(row["lon"]) and pd.notnull(row["meetwaarde"])
    ]
    HeatMap(
        heat_data,
        radius=12,
        max_zoom=13,
        gradient={0.2: 'lightblue', 0.4: 'deepskyblue', 0.6: 'blue', 0.8: 'navy', 1.0: 'darkblue'}
    ).add_to(m)

    st_folium(m, width=800, height=600)


with col2:
    df_avg = df_filtered_param.groupby("year")["meetwaarde"].mean().reset_index()

    chart = alt.Chart(df_avg).mark_line(point=True).encode(
        x=alt.X('year:O', title='Jaar'),
        y=alt.Y('meetwaarde:Q', title=f'Gemiddelde {selected_param}'),
        tooltip=['year', 'meetwaarde']
    ).properties(
        width=350,
        height=600,
        title=f"Gemiddelde {selected_param} per jaar"
    )

    highlight = alt.Chart(df_avg[df_avg["year"] == selected_year]).mark_point(color='red', size=100).encode(
        x='year:O',
        y='meetwaarde:Q'
    )

    st.altair_chart(chart + highlight)
