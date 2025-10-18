import streamlit as st
import pandas as pd
import pydeck as pdk

@st.cache_data
def load_data():
    return pd.read_excel("../links.xlsx", engine="openpyxl")

df = load_data()
df = df.dropna(subset=['longitude_A', 'latitude_A', 'longitude_B', 'latitude_B'])

available_operators = df['Operator'].dropna().unique().tolist()

# Defaultowi operatorzy
default_operators = [
    "ORANGE POLSKA S.A.",
    "P4 Sp. z o.o.",
    "T-Mobile Polska S.A.",
    "Towerlink Poland Sp. z o.o."
]

# Kolory dla wskazanych operatorów
operator_colors = {
    "ORANGE POLSKA S.A.": [255, 121, 0],       # Orange
    "P4 Sp. z o.o.": [75, 0, 130],            # P4
    "T-Mobile Polska S.A.": [226, 0, 116],    # T-Mobile
    "Towerlink Poland Sp. z o.o.": [0, 166, 81] # Towerlink
}

# Stały kolor dla pozostałych operatorów
default_color = [180, 180, 180]

st.title("Mapa linii radiowych MW")
selected_operators = st.multiselect("Wybierz operatorów", available_operators, default=default_operators)

df_filtered = df[df['Operator'].isin(selected_operators)]

lines = []
for _, row in df_filtered.iterrows():
    color = operator_colors.get(row['Operator'], default_color)
    tooltip = (
        f"Operator: {row['Operator']}\n"
        f"TX: {row['Miejscowość Tx']} ({row['H_ant_Tx [m npt]']} m)\n"
        f"RX: {row['Miejscowość Rx']} ({row['H_ant_Rx [m npt]']} m)\n"
        f"Freq: {row['freq_B']} GHz\n"
        f"Kanał: {row['Nr_kan_B']}\n"
        f"Modulacja: {row['Rodz_modu-lacji']}\n"
        f"Pojemność: {row['Przepływność [Mb/s]']} Mb/s"
    )
    lines.append({
        "start": [row['longitude_A'], row['latitude_A']],
        "end": [row['longitude_B'], row['latitude_B']],
        "color": color,
        "tooltip": tooltip
    })

layer = pdk.Layer(
    "LineLayer",
    data=lines,
    get_source_position="start",
    get_target_position="end",
    get_color="color",
    get_width=2,
    pickable=True,
    auto_highlight=True
)

osm_layer = pdk.Layer(
    "TileLayer",
    data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    min_zoom=0,
    max_zoom=19,
    tile_size=256
)

mid_lat = df_filtered['latitude_A'].mean()
mid_lon = df_filtered['longitude_A'].mean()
view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=6)

# Wymuszenie kursora crosshair
st.markdown("<style>canvas {cursor: crosshair !important;}</style>", unsafe_allow_html=True)

st.pydeck_chart(pdk.Deck(
    initial_view_state=view_state,
    layers=[osm_layer, layer],
    tooltip={"text": "{tooltip}"}

))
