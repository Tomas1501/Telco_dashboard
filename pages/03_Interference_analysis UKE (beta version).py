# -*- coding: utf-8 -*-
# Analiza interferencji & rekomendacja kanału – wersja uproszczona (bez sidebaru i debug)
# Funkcje:
#  - Interferencja międzyplanowa w obrębie pasma (np. 75/85A250 ↔ 75/85A125)
#  - ACI z nakładania widm (prostokątne maski) + quasi-cochannel: overlap >= 0.8 * min(BW_i, BW_c)
#  - Kara za quasi-współkanał +60 dB (zamiast +100)
#  - Dwukierunkowo: wynik = max(IN_fwd, IN_rev) i odpowiadająca degradacja
#  - Rekomendacja kanału z kryterium UKE: Degradacja_max ≤ 1 dB
#  - Wykres łączony: słupki I/N max (lewa oś) + linia Degradacja max (prawa oś) + linia progu 1 dB
#  - Kolor słupków: zielony (UKE_ok) / czerwony (naruszenie)
#  - Tabela rozszerzona o Degr_fwd/Degr_rev/Degr_max i UKE_ok + eksport CSV
#  - Mapa pydeck + TOP interferenci dla gorszego kierunku
#  - W UI pozostaje wybór pasma, planu ocenianego, kanału (opcjonalnie) i planów interferencyjnych
#
# Uproszczenia zgodnie z prośbą:
#  - Brak sidebaru i debug
#  - Brak wejścia NF (stałe DEFAULT_NF_DB)
#  - Brak wejścia progu UKE (stałe UKE_THR_DB=1.0)
#  - Brak wejścia promienia – używamy domyślnego per pasmo (DEFAULT_RADIUS_BY_BAND_KM)

import os
import io
import re
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

hide_toolbar = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_toolbar, unsafe_allow_html=True)

st.set_page_config(page_title="Analiza interferencji – rekomendacja kanału", layout="wide")

# =========================
# KONFIG – zmień ścieżkę na właściwy XLSX u siebie
# =========================
from pathlib import Path

# Znajdź katalog główny repo (rodzic folderu pages)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_XLSX = ROOT_DIR / "linie_radiowe_stan_na_2025-09-25.xlsx"

# Stałe modelu
DEFAULT_EIRP_DBM = 55.0
DEFAULT_EIRP_DBM_EBAND = 51.0
DEFAULT_DIAM_M = 0.6
DEFAULT_NF_DB = 9.5            # Noise Figure do I/N (stałe)
UKE_THR_DB = 1.0               # próg degradacji wg UKE [dB]

# Prefiksy promienia (km) per pasmo
DEFAULT_RADIUS_BY_BAND_KM = {
    7: 80, 13: 50, 18: 45, 23: 40, 32: 30, 38: 25, 80: 15  # E‑Band => 80
}

# Parametry propagacji / anten
R_EARTH = 6371.0
XPD_DB = 30.0  # XPD dla V/H

# Dystanse reuse per pasmo (km)
REUSE_BY_BAND = {7: 60, 13: 40, 18: 30, 23: 25, 32: 15, 38: 10, 80: 2}

# =========================
# Parser współrzędnych (tolerancyjny)
# =========================
def dms_tolerant_to_dd(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    s_norm = (
        s.replace("″", '"').replace("”", '"').replace("“", '"')
         .replace("ʼ", "'").replace("’", "'").replace("′", "'")
         .replace("''", '"')
    )
    s_norm = re.sub(r"\s+([NSEW])\s*$", r"\1", s_norm, flags=re.IGNORECASE)
    s_norm = re.sub(r"^\s*([NSEW])\s+", r"\1", s_norm, flags=re.IGNORECASE)
    s_up = s_norm.upper()

    m = re.match(r"^\s*(\d+)\s*([EWNS])\s*(\d+)?'?\s*(\d+(?:[.,]\d+)?)?\"?\s*$", s_up)
    if m:
        deg = float(m.group(1)); hemi = m.group(2)
        minutes = float(m.group(3) or 0.0)
        seconds = float((m.group(4) or "0").replace(",", "."))
        dd = deg + minutes/60.0 + seconds/3600.0
        if hemi in ("W", "S"):
            dd = -dd
        return dd

    m = re.match(r"^\s*(\d+)[-\s:]+(\d+)[-\s:]+(\d+(?:[.,]\d+)?)([NSEW])\s*$", s_up)
    if m:
        d = float(m.group(1)); mnt = float(m.group(2))
        sec = float(m.group(3).replace(",", "."))
        hemi = m.group(4)
        dd = d + mnt/60.0 + sec/3600.0
        if hemi in ("W", "S"):
            dd = -dd
        return dd

    m = re.match(r"^\s*(\d+)[-\s:]+(\d+)([NSEW])\s*$", s_up)
    if m:
        d = float(m.group(1)); mnt = float(m.group(2)); hemi = m.group(3)
        dd = d + mnt/60.0
        if hemi in ("W", "S"):
            dd = -dd
        return dd

    m = re.match(r"^\s*(\d+)([NSEW])\s*$", s_up)
    if m:
        d = float(m.group(1)); hemi = m.group(2)
        dd = d
        if hemi in ("W", "S"):
            dd = -dd
        return dd

    try:
        return float(s_up.replace(",", "."))
    except Exception:
        return np.nan

# =========================
# GEOMETRIA I MODEL
# =========================
def hav_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R_EARTH*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.rad2deg(np.arctan2(x, y)) + 360) % 360

def hpbw_deg(f_ghz, d_m, default=1.0):
    d = float(d_m) if d_m and float(d_m) > 0 else 1.0
    f = np.asarray(f_ghz, dtype=float)
    f = np.maximum(f, 1e-9)
    lam = 0.3 / f
    hpbw = 70.0 * lam / d
    return float(hpbw) if np.isscalar(f_ghz) else hpbw

def offaxis_parabolic(phi_deg, hpbw, amax=55.0):
    phi = np.asarray(phi_deg, dtype=float)
    h = np.asarray(hpbw, dtype=float)
    h = np.maximum(h, 0.5)
    att = 12.0*(np.abs(phi)/h)**2
    return np.minimum(att, amax)
def est_d_from_gain_db(g_dbi: float, f_ghz: float, eta: float = 0.6) -> float:
    """Szacuje średnicę anteny parabolicznej ze zysku (dBi) i częstotliwości."""
    try:
        if g_dbi is None or f_ghz is None or g_dbi <= 0 or f_ghz <= 0:
            return float("nan")
        lam = 0.3 / float(f_ghz)  # [m]
        g_lin = 10 ** (float(g_dbi) / 10.0)
        # G ≈ η * (π D / λ)^2  =>  D ≈ (λ/π) * sqrt(G/η)
        D = lam / math.pi * math.sqrt(g_lin / max(eta, 0.3))
        return float(D)
    except Exception:
        return float("nan")

def hpbw_from_gain(g_dbi_arr, f_ghz_arr, fallback_d_m: float = 0.3):
    """HPBW (deg) z oszacowanej średnicy; fallback do D=fallback_d_m (np. 0.3 m dla E-band)."""
    g = np.asarray(g_dbi_arr, dtype=float)
    f = np.asarray(f_ghz_arr, dtype=float)
    D = np.vectorize(est_d_from_gain_db)(g, f)
    # gdy brak danych – użyj średnicy domyślnej
    D = np.where(np.isfinite(D), D, fallback_d_m)
    lam = 0.3 / np.maximum(f, 1e-9)
    hpbw = 70.0 * lam / np.maximum(D, 0.05)
    return hpbw

def fspl_db(d_km: float, f_ghz: float) -> float:
    d_km = max(d_km, 0.001)
    f_ghz = max(f_ghz, 0.1)
    return 92.45 + 20*math.log10(d_km) + 20*math.log10(f_ghz)

def reuse_for_band(band: int) -> float:
    return REUSE_BY_BAND.get(int(band), 20.0)

def noise_floor_dbm(bw_mhz: float, nf_db: float = DEFAULT_NF_DB) -> float:
    if not bw_mhz or pd.isna(bw_mhz):
        return -999.0
    bw_hz = float(bw_mhz) * 1e6
    return -174.0 + 10.0*math.log10(bw_hz) + float(nf_db)

# =========================
# ŁADOWANIE DANYCH + standaryzacja
# =========================
def guess_band_from_f(f_ghz: float) -> int:
    if f_ghz is None or (isinstance(f_ghz, float) and np.isnan(f_ghz)):
        return np.nan
    f = float(f_ghz)
    if 70.0 <= f <= 86.0:
        return 80
    candidates = np.array([7, 13, 18, 23, 32, 38], dtype=float)
    return int(candidates[np.argmin(np.abs(candidates - f))])

def pick(cols, options):
    for opt in options:
        for c in cols:
            if str(c).strip().lower() == opt.lower():
                return c
    for opt in options:
        for c in cols:
            if opt.lower() in str(c).strip().lower():
                return c
    return None

def _load_links_no_cache(path_str: str):
    info = {"sheet_names": [], "header_idx": None, "raw_columns": [], "nan_counts": {}}
    if not os.path.exists(path_str):
        return pd.DataFrame(), info, f"Plik nie istnieje: {path_str}"
    try:
        with open(path_str, "rb") as f:
            file_bytes = f.read()
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        info["sheet_names"] = list(xls.sheet_names)
        best = None
        for sh in xls.sheet_names:
            df0 = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sh, dtype=str, engine="openpyxl")
            if any("Dl_geo_Tx" in str(c) for c in df0.columns):
                best = df0
                break
        if best is None:
            best = pd.read_excel(io.BytesIO(file_bytes), sheet_name=xls.sheet_names[0], dtype=str, engine="openpyxl")

        raw = best.dropna(how="all").copy()
        info["raw_columns"] = [str(c) for c in raw.columns]
        header_idx = None
        for i in range(min(10, len(raw))):
            row = raw.iloc[i].astype(str).tolist()
            if any("Dl_geo_Tx" in v for v in row):
                header_idx = i
                break
        info["header_idx"] = header_idx
        if header_idx is not None:
            new_header = raw.iloc[header_idx]
            df = raw.iloc[header_idx+1:].copy()
            df.columns = new_header
        else:
            df = raw.copy()

        cols = list(df.columns)
        col_tx_lon = pick(cols, ["Dl_geo_Tx"])
        col_tx_lat = pick(cols, ["Sz_geo_Tx"])
        col_rx_lon = pick(cols, ["Dl_geo_Rx"])
        col_rx_lat = pick(cols, ["Sz_geo_Rx"])
        col_f      = pick(cols, ["f [GHz]", "f"])
        col_plan   = pick(cols, ["Symbol_planu", "Symbol planu"])
        col_bw     = pick(cols, ["Szer_kan [MHz]", "Szer_kan", "Szer.kan [MHz]"])
        col_ch     = pick(cols, ["Nr_kan", "Nr_kanal", "Nr kan"])
        col_pol    = pick(cols, ["Polaryzacja"])
        col_eirp   = pick(cols, ["EIRP [dBm]", "EIRP"])
        col_tx_g   = pick(cols, ["Zysk_ant_Tx [dBi]", "Zysk_ant_Tx"])
        col_rx_g   = pick(cols, ["Zysk_ant_Rx [dBi]", "Zysk_ant_Rx"])

        need = [col_tx_lon, col_tx_lat, col_rx_lon, col_rx_lat, col_f, col_plan, col_ch, col_pol]
        if any(c is None for c in need):
            return (pd.DataFrame(), info,
                    "Brakuje oczekiwanych kolumn: Dl_geo_Tx / Sz_geo_Tx / Dl_geo_Rx / Sz_geo_Rx / f [GHz] / "
                    "Symbol_planu / Nr_kan / Polaryzacja.")

        links = df[[c for c in [col_tx_lon, col_tx_lat, col_rx_lon, col_rx_lat, col_f, col_plan, col_bw, col_ch, col_pol,
                                col_eirp, col_tx_g, col_rx_g] if c is not None]].copy()
        links.columns = ["tx_lon","tx_lat","rx_lon","rx_lat","f_ghz","plan","bw_mhz","chan","pol",
                         "eirp_dbm","tx_gain_dbi","rx_gain_dbi"]

        # numeryczne
        links["f_ghz"] = pd.to_numeric(links["f_ghz"].astype(str).str.replace(",", "."), errors="coerce")
        for c in ["bw_mhz","eirp_dbm","tx_gain_dbi","rx_gain_dbi"]:
            if c in links.columns:
                links[c] = pd.to_numeric(links[c].astype(str).str.replace(",", "."), errors="coerce")

        # kanały: etykieta, liczba, Hi/Lo z apostrofu na końcu
        links["chan_label"] = links["chan"].astype(str).str.strip()
        links["chan_num"]   = pd.to_numeric(links["chan_label"].str.extract(r"(\-?\d+)")[0], errors="coerce")
        links["chan_is_hi"] = links["chan_label"].str.contains(r"['′]\s*$")

        # polaryzacja do H/V
        links["pol"] = links["pol"].astype(str).str.upper().str.strip()
        links["pol"] = np.where(links["pol"].str.startswith("H"), "H",
                         np.where(links["pol"].str.startswith("V"), "V", links["pol"]))

        # współrzędne → DD
        links["tx_lon_dd"] = links["tx_lon"].apply(dms_tolerant_to_dd)
        links["tx_lat_dd"] = links["tx_lat"].apply(dms_tolerant_to_dd)
        links["rx_lon_dd"] = links["rx_lon"].apply(dms_tolerant_to_dd)
        links["rx_lat_dd"] = links["rx_lat"].apply(dms_tolerant_to_dd)

    
        # EIRP domyślne per pasmo – NIE mieszamy z zyskiem anteny
        if "eirp_dbm" in links.columns:
            links["eirp_dbm"] = pd.to_numeric(links["eirp_dbm"], errors="coerce")
        else:
            links["eirp_dbm"] = np.nan

        links["eirp_dbm"] = np.where(
            links["eirp_dbm"].notna(), links["eirp_dbm"],
            np.where(links["band"].eq(80), DEFAULT_EIRP_DBM_EBAND, DEFAULT_EIRP_DBM)
        )
        
        # band (E‑Band => 80)
        links["band"] = links["f_ghz"].apply(guess_band_from_f)

        # EIRP domyślne per pasmo – NIE mieszamy z zyskiem anteny (ustaw raz, po band)
        if "eirp_dbm" in links.columns:
            links["eirp_dbm"] = pd.to_numeric(links["eirp_dbm"], errors="coerce")
        else:
            links["eirp_dbm"] = np.nan

        links["eirp_dbm"] = np.where(
            links["eirp_dbm"].notna(), links["eirp_dbm"],
            np.where(links["band"].eq(80), DEFAULT_EIRP_DBM_EBAND, DEFAULT_EIRP_DBM)
        )

        # kluczowe kolumny
        key_cols = ["tx_lon_dd","tx_lat_dd","rx_lon_dd","rx_lat_dd","f_ghz","plan","chan_num","band"]
        info["nan_counts"] = {c: int(links[c].isna().sum()) for c in key_cols}
        links = links.dropna(subset=key_cols).copy()

        # azymut TX istniejących łączy
        links["tx_bearing"] = links.apply(
            lambda r: bearing_deg(r["tx_lat_dd"], r["tx_lon_dd"], r["rx_lat_dd"], r["rx_lon_dd"]),
            axis=1
        )

        return links.reset_index(drop=True), info, None
    except Exception as e:
        return pd.DataFrame(), info, f"Wyjątek przy czytaniu pliku: {type(e).__name__}: {e}"

@st.cache_data(show_spinner=False)
def load_links_cached(path_str: str, mtime: float):
    df, _, _ = _load_links_no_cache(path_str)
    return df

def load_links_with_info(path_str: str):
    return _load_links_no_cache(path_str)

# =========================
# Lookup częstotliwości środkowych kanałów
# =========================
def build_fc_lookup(df: pd.DataFrame) -> dict:
    g = (
        df.dropna(subset=["chan_num", "f_ghz"])
          .groupby(["band", "plan", "chan_is_hi", "chan_num"])["f_ghz"]
          .median()
    )
    return { (int(k[0]), str(k[1]), bool(k[2]), int(k[3])): float(v) for k, v in g.items() }

def fc_75_85_A250_lo(ch: int) -> float:
    """Kanały Lo (GHz) dla planu 75/85A250 — ch=11 => 73.75, ch=12 => 74.00, itd."""
    return 71.0 + 0.25 * int(ch)


def get_fc(fc_lookup: dict, band: int, plan: str, is_hi: bool, ch: int, fallback_f_ghz: float) -> float:
    # Priorytet: twarda mapa dla 75/85A250 w E-band
    if int(band) == 80 and str(plan).strip().upper().startswith("75/85A250"):
        f_lo = fc_75_85_A250_lo(ch)
        return f_lo + (10.0 if is_hi else 0.0)

    # Dalej: lookup z danych (jak dotąd)
    key = (int(band), str(plan), bool(is_hi), int(ch))
    if key in fc_lookup:
        return fc_lookup[key]
    cand = [v for (b, p, h, c), v in fc_lookup.items() if b == int(band) and p == str(plan) and h == bool(is_hi)]
    if cand:
        return float(np.median(cand))
    return float(fallback_f_ghz)

# =========================
# ACI z nakładania widm (prostokąty)
# =========================
# ZAMIANA funkcji spectral_adj_db_vec na wersję maskową
def spectral_adj_db_vec(f_i_ghz: np.ndarray, bw_i_mhz: np.ndarray,
                        f_c_ghz: float, bw_c_mhz: float,
                        adj_cap_db: float = 60.0) -> np.ndarray:
    """
    ACI wg prostej maski:
    - Jeśli jest nakładanie (overlap>0): Adj = -10*log10(overlap/BW_i) (jak dotąd).
    - Jeśli pasma się nie nakładają:
        separacja s = |fi-fc| / ((BW_i + BW_c)/2000)  [w jednostkach "połowy sumy BW"]
        s=1  -> Adj ≈ 35 dB
        s=2  -> Adj ≈ 50 dB
        s>=3 -> 60 dB (cap)
      z interpolacją liniową między punktami.
    """
    df_mhz = np.abs(f_i_ghz - float(f_c_ghz)) * 1000.0
    bw_i = np.maximum(bw_i_mhz.astype(float), 1e-6)
    bw_c = max(float(bw_c_mhz), 1e-6)

    # overlap prostokątów
    overlap = np.maximum(0.0, bw_i/2.0 + bw_c/2.0 - df_mhz)
    frac = np.clip(overlap / bw_i, 1e-6, 1.0)
    adj_overlap = -10.0 * np.log10(frac)

    half_sum = (bw_i + bw_c) / 2.0
    s = np.where(half_sum > 0, df_mhz / half_sum, np.inf)

    adj_mask = np.full_like(s, adj_cap_db, dtype=float)
    # 1×BW -> 35 dB, 2×BW -> 50 dB
    adj_1, adj_2 = 35.0, 50.0
    between_1_2 = (s >= 1.0) & (s < 2.0)
    adj_mask[between_1_2] = adj_1 + (adj_2 - adj_1) * (s[between_1_2] - 1.0) / 1.0
    below_1 = (s > 0.0) & (s < 1.0)
    adj_mask[below_1] = np.clip(30.0 + 5.0 * s[below_1], 30.0, adj_1)

    adj = np.where(overlap > 0.0, adj_overlap, adj_mask)
    return np.clip(adj, 0.0, adj_cap_db)
# --- ACIR(Δf) schodkowy (wg odległości od kanału w MHz) ---
def acir_step_db(delta_f_mhz: np.ndarray,
                 th_125=45.0, th_250=55.0, th_375=60.0, th_500=65.0, th_ge=70.0) -> np.ndarray:
    d = np.abs(delta_f_mhz.astype(float))
    out = np.full_like(d, th_ge, dtype=float)
    out[d < 1.0] = 0.0                   # współkanał
    out[(d >= 1.0) & (d <= 125.0)] = th_125
    out[(d > 125.0) & (d <= 250.0)] = th_250
    out[(d > 250.0) & (d <= 375.0)] = th_375
    out[(d > 375.0) & (d <= 500.0)] = th_500
    return out

def aci_combine_min(acir_db: np.ndarray, overlap_db: np.ndarray) -> np.ndarray:
    """
    Konserwatywne łączenie: MIN(ACIR, OverlapMask)
    - mniejsza z tłumien (w dB) => większa interferencja (bezpiecznie pod UKE)
    """
    return np.minimum(acir_db, overlap_db)

# =========================
# Prefiltr promienia
# =========================
def filter_by_radius(db: pd.DataFrame, rx_lat: float, rx_lon: float, radius_km: float) -> pd.DataFrame:
    if db.empty:
        return db
    D = hav_km(db["tx_lat_dd"].to_numpy(), db["tx_lon_dd"].to_numpy(),
               np.full(len(db), rx_lat), np.full(len(db), rx_lon))
    return db.loc[D <= float(radius_km)].copy()

# =========================
# SCORING – jedno- i dwukierunkowo (z BW i widmem)
# =========================
def score_one_side(candidate, db_plan: pd.DataFrame, plan_channels: list[int],
                   bw_mhz_plan: float, nf_db_ui: float,
                   fc_lookup: dict, plan_sel: str, is_hi: bool) -> pd.DataFrame:
    """
    candidate: latA, lonA, latB, lonB, pol, band (+ rx_hpbw_mode, rx_gain_dbi, rx_diam_m)
    db_plan: interferenci (dowolne plany w obrębie pasma), właściwy kierunek Lo/Hi
    plan_channels: lista ocenianych kanałów 'plan_sel' (None -> wszystkie dostępne)
    bw_mhz_plan: szerokość kanału RX kandydata
    nf_db_ui: Noise Figure
    fc_lookup: lookup f_center dla [band, plan, Lo/Hi, channel]
    plan_sel, is_hi: plan oceniany i jego połówka (Lo=False, Hi=True)
    """
    tx_lat, tx_lon = candidate["latA"], candidate["lonA"]
    rx_lat, rx_lon = candidate["latB"], candidate["lonB"]
    pol_c = candidate["pol"]
    band  = candidate["band"]
    rx_brg = bearing_deg(rx_lat, rx_lon, tx_lat, tx_lon)

    # Pusty zbiór -> zwróć pusty szkielet z oczekiwanymi kolumnami
    if db_plan.empty:
        return pd.DataFrame(columns=["channel", "I_dBm", "IN_dB", "N_dbm"])

    # Wektory interferentów
    txi_lat = db_plan["tx_lat_dd"].to_numpy()
    txi_lon = db_plan["tx_lon_dd"].to_numpy()
    eirp_i  = db_plan["eirp_dbm"].to_numpy()
    pol_i   = db_plan["pol"].astype(str).str.upper().str[0].to_numpy()
    f_i     = db_plan["f_ghz"].to_numpy()
    bw_i    = pd.to_numeric(
                 db_plan.get("bw_mhz", pd.Series(np.nan, index=db_plan.index)),
                 errors="coerce"
             ).fillna(
                 pd.to_numeric(db_plan["bw_mhz"], errors="coerce").median()
                 if "bw_mhz" in db_plan.columns else bw_mhz_plan
             ).to_numpy()
    tx_brg_i = db_plan["tx_bearing"].to_numpy()

    # Geometria i kąty
    D    = hav_km(txi_lat, txi_lon, np.full_like(txi_lat, rx_lat), np.full_like(txi_lon, rx_lon))
    be_tx = bearing_deg(txi_lat, txi_lon, np.full_like(txi_lat, rx_lat), np.full_like(txi_lon, rx_lon))
    be_rx = bearing_deg(np.full_like(txi_lat, rx_lat), np.full_like(txi_lon, rx_lon), txi_lat, txi_lon)
    ang_tx = np.abs((tx_brg_i - be_tx + 180) % 360 - 180)

    # HPBW TX z zysku (fallback: 0.3 m w E-band)
    tx_gain_i = pd.to_numeric(db_plan.get("tx_gain_dbi", pd.Series(np.nan, index=db_plan.index)),
                              errors="coerce").to_numpy()
    tx_hpbw_i = hpbw_from_gain(tx_gain_i, f_i, fallback_d_m=(0.3 if int(band) == 80 else DEFAULT_DIAM_M))

    # Straty ścieżki i XPD
    FSPL    = np.array([fspl_db(d, fi) for d, fi in zip(D, f_i)])
    XPD_arr = np.where(pol_i == pol_c, 0.0, XPD_DB)

    # Szum odbiornika
    N_dbm = noise_floor_dbm(bw_mhz_plan, nf_db_ui)

    # Zakres kanałów do oceny
    if plan_channels is None:
        chans_known = sorted(set(
            db_plan.loc[db_plan["plan"] == plan_sel, "chan_num"].dropna().astype(int).tolist()
        ))
        channels = chans_known if chans_known else sorted(set(db_plan["chan_num"].dropna().astype(int).tolist()))
    else:
        channels = sorted(set(int(c) for c in plan_channels))

    # Parametry RX kandydata (z UI)
    rx_mode   = str(candidate.get("rx_hpbw_mode", "gain")).lower()  # "gain" | "diameter"
    rx_gain_d = float(candidate.get("rx_gain_dbi", 46.0))           # sensowny default dla E-band
    rx_diam_m = float(candidate.get("rx_diam_m", 0.31))             # ~0.31 m ≈ 46 dBi @ 80 GHz

    rows = []
    reuse = reuse_for_band(band)

    for k in channels:
        # Częstotliwość środkowa kanału
        f_c = get_fc(fc_lookup, band=band, plan=plan_sel, is_hi=is_hi, ch=k, fallback_f_ghz=float(band))

        # HPBW RX wg trybu (z zysku lub średnicy)
        if rx_mode == "gain":
            rx_hpbw = hpbw_from_gain(np.array([rx_gain_d]), np.array([f_c]),
                                     fallback_d_m=(0.3 if int(band) == 80 else DEFAULT_DIAM_M))[0]
        else:
            rx_hpbw = hpbw_deg(f_c, rx_diam_m)

        # Kąty RX
        ang_rx = np.abs((rx_brg - be_rx + 180) % 360 - 180)

        # Tłumienia pozaosiowe
        att_tx = offaxis_parabolic(ang_tx, tx_hpbw_i)
        att_rx = offaxis_parabolic(ang_rx, rx_hpbw)

        # --- PSD + integracja widma + ACIR = MIN(ACIR, Overlap) ---
        # 1) Parametry pasm
        B_rx_Hz = float(bw_mhz_plan) * 1e6
        df_mhz  = (f_i - f_c) * 1000.0

        # Overlap (MHz -> Hz)
        ov_mhz  = np.maximum(0.0, bw_i/2.0 + bw_mhz_plan/2.0 - np.abs(df_mhz))
        B_ov_Hz = ov_mhz * 1e6

        # 2) ACI: Overlap + ACIR(Δf) i konserwatywne MIN
        Adj_overlap = spectral_adj_db_vec(f_i, bw_i, f_c, bw_mhz_plan)
        Adj_acir    = acir_step_db(df_mhz, th_125=45.0, th_250=55.0, th_375=60.0, th_500=65.0, th_ge=70.0)
        Adj         = aci_combine_min(Adj_acir, Adj_overlap)

        # 3) PSD (dBm/Hz) i integracja po BW
        eirp_dens_dBm_per_Hz = eirp_i - 10.0*np.log10(np.maximum(bw_i, 1e-6) * 1e6)
        B_int_Hz = np.where(B_ov_Hz > 0.0, B_ov_Hz, B_rx_Hz)

        # 4) Poziom I od pojedynczych interferentów (dBm)
        I_each = (eirp_dens_dBm_per_Hz
                  - Adj
                  + 10.0*np.log10(B_int_Hz)
                  - FSPL
                  - att_tx
                  - att_rx
                  - XPD_arr)

        # 5) Quasi‑cochannel penalty (ostrzej: +100 dB)
        min_bw   = np.minimum(bw_i, bw_mhz_plan)
        quasi_co = B_ov_Hz >= (0.8 * min_bw * 1e6)
        aligned  = (ang_tx < 3*hpbw_deg(f_i, DEFAULT_DIAM_M)) & (ang_rx < 3*rx_hpbw)
        I_each[(D < reuse) & aligned & quasi_co] += 100.0

        # 6) Suma energetyczna + I/N
        Ilin  = np.sum(10.0**(I_each/10.0))
        I_dBm = 10.0*np.log10(max(Ilin, 1e-30))
        IN_dB = I_dBm - N_dbm

        # >>> TU BYŁ BRAK <<<  — dodajemy wiersz do rows
        rows.append({
            "channel": int(k),
            "I_dBm": float(I_dBm),
            "IN_dB": float(IN_dB),
            "N_dbm": float(N_dbm)
        })

    # Zwróć uporządkowaną tabelę; bezpiecznie obsłuż pusty przypadek
    if not rows:
        return pd.DataFrame(columns=["channel", "I_dBm", "IN_dB", "N_dbm"])

    df = (pd.DataFrame(rows, columns=["channel", "I_dBm", "IN_dB", "N_dbm"])
            .sort_values("IN_dB", kind="mergesort")
            .reset_index(drop=True))
    return df

def score_bidir(candidate, db_lo: pd.DataFrame, db_hi: pd.DataFrame,
                f_lo_ghz: float, f_hi_ghz: float,
                bw_mhz_plan: float, plan_sel: str,
                fc_lookup: dict, nf_db_ui: float,
                plan_channels: list[int] | None = None) -> pd.DataFrame:

    # Wyznacz wspólne kanały dla planu ocenianego
    chans_lo = set(db_lo.loc[db_lo["plan"] == plan_sel, "chan_num"].dropna().astype(int).tolist())
    chans_hi = set(db_hi.loc[db_hi["plan"] == plan_sel, "chan_num"].dropna().astype(int).tolist())
    chans_common = sorted(chans_lo & chans_hi)
    if not chans_common:
        chans_common = sorted(chans_lo or chans_hi)
    if plan_channels is not None:
        chans_common = sorted(set(chans_common) & set(plan_channels))
    if not chans_common:
        return pd.DataFrame(columns=["channel","I_fwd_dBm","IN_fwd_dB","I_rev_dBm","IN_rev_dB","IN_max_dB","N_dbm"])

    # A→B (Lo)
    cand_lo = dict(candidate)
    cand_lo["f_ghz"] = float(f_lo_ghz)
    res_lo = score_one_side(cand_lo, db_lo, chans_common, bw_mhz_plan, nf_db_ui=nf_db_ui,
                            fc_lookup=fc_lookup, plan_sel=plan_sel, is_hi=False
                           ).rename(columns={"I_dBm":"I_fwd_dBm","IN_dB":"IN_fwd_dB","N_dbm":"N_fwd_dbm"})

    # B→A (Hi)
    cand_hi = dict(candidate)
    cand_hi["latA"], cand_hi["lonA"], cand_hi["latB"], cand_hi["lonB"] = candidate["latB"], candidate["lonB"], candidate["latA"], candidate["lonA"]
    cand_hi["f_ghz"] = float(f_hi_ghz)
    res_hi = score_one_side(cand_hi, db_hi, chans_common, bw_mhz_plan, nf_db_ui=nf_db_ui,
                            fc_lookup=fc_lookup, plan_sel=plan_sel, is_hi=True
                           ).rename(columns={"I_dBm":"I_rev_dBm","IN_dB":"IN_rev_dB","N_dbm":"N_rev_dbm"})

    res = pd.merge(res_lo, res_hi, on="channel", how="outer")
    res["N_dbm"] = res[["N_fwd_dbm","N_rev_dbm"]].mean(axis=1, skipna=True)
    res["IN_max_dB"] = res[["IN_fwd_dB","IN_rev_dB"]].max(axis=1)

    # --- Degradacja z I/N (per kierunek) + max dwukierunkowo ---
    def _degr_from_in_db(in_db_col: pd.Series) -> pd.Series:
        in_lin = 10 ** (pd.to_numeric(in_db_col, errors="coerce") / 10.0)
        return 10.0 * np.log10(1.0 + in_lin)

    res["Degr_fwd_dB"] = _degr_from_in_db(res["IN_fwd_dB"])
    res["Degr_rev_dB"] = _degr_from_in_db(res["IN_rev_dB"])
    res["Degr_max_dB"] = res[["Degr_fwd_dB", "Degr_rev_dB"]].max(axis=1)
    res["UKE_ok"] = res["Degr_max_dB"] <= UKE_THR_DB

    return res.sort_values("IN_max_dB")

# =========================
# MAPKA (pydeck)
# =========================
def make_map(latA, lonA, latB, lonB, top_df: pd.DataFrame, worst_dir_label: str, radius_km: float):
    # punkty A/B
    df_nodes = pd.DataFrame([
        {"name": "A (TX/RX)", "lon": lonA, "lat": latA, "color": [0, 120, 255]},
        {"name": "B (TX/RX)", "lon": lonB, "lat": latB, "color": [0, 120, 255]},
    ])
    layer_nodes = pdk.Layer(
        "ScatterplotLayer",
        df_nodes,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=80,
        pickable=True,
    )
    # linia A↔B
    df_link = pd.DataFrame([{"from_lon": lonA, "from_lat": latA, "to_lon": lonB, "to_lat": latB}])
    layer_link = pdk.Layer(
        "GreatCircleLayer",
        df_link,
        get_source_position='[from_lon, from_lat]',
        get_target_position='[to_lon, to_lat]',
        get_stroke_width=3,
        get_source_color=[0, 120, 255],
        get_target_color=[0, 120, 255],
    )
    # interferenci (od TX interferenta do RX kandydata w gorszym kierunku)
    layer_i = pdk.Layer(
        "GreatCircleLayer",
        top_df,
        get_source_position='[tx_lon, tx_lat]',
        get_target_position='[rx_lon, rx_lat]',
        get_stroke_width=2,
        get_source_color=[255, 90, 0],
        get_target_color=[255, 90, 0],
        pickable=True,
    )
    # centrum i zoom
    center_lon = (lonA + lonB) / 2.0
    center_lat = (latA + latB) / 2.0
    zoom = 9 if radius_km <= 15 else (8 if radius_km <= 30 else 7)
    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, bearing=0, pitch=0)
    r = pdk.Deck(
        layers=[layer_nodes, layer_link, layer_i],
        initial_view_state=view,
        tooltip={"text": "{name}"},
        map_style="light"
    )
    return r

# =========================
# UI
# =========================
st.title("Analiza interferencji & rekomendacja kanału (MW) – międzyplanowa")

# Wczytywanie
if not os.path.exists(DEFAULT_DATA_XLSX):
    st.error(f"Plik XLSX nie istnieje: {DEFAULT_DATA_XLSX}\nZmień stałą DEFAULT_DATA_XLSX w pliku pages/app.py.")
    st.stop()

mtime = DEFAULT_DATA_XLSX.stat().st_mtime
links_df = load_links_cached(str(DEFAULT_DATA_XLSX), mtime)

if links_df is None or links_df.empty:
    links_df, info_dbg, err_dbg = load_links_with_info(DEFAULT_DATA_XLSX)
    if err_dbg:
        st.error(err_dbg)
    if links_df.empty:
        st.error("Brak danych po wczytaniu.")
        st.stop()

# Wybór pasma -> plan oceniany -> kanał (opcjonalnie)
bands = sorted(links_df["band"].dropna().astype(int).unique().tolist())
band_labels = {b: ("E‑Band (70/80 GHz)" if b == 80 else f"{b} GHz") for b in bands}
band_options = [band_labels[b] for b in bands]
default_band_index = band_options.index(band_labels[80]) if 80 in bands else 0

st.subheader("Wybór pasma / planu ocenianego / planów interferencyjnych")
c1, c2, c3 = st.columns([1, 2, 2])
with c1:
    band_label_sel = st.selectbox("Pasmo", band_options, index=default_band_index)
    band_sel = [b for b, lab in band_labels.items() if lab == band_label_sel][0]
with c2:
    plans_for_band = sorted(links_df.loc[links_df["band"]==band_sel, "plan"].dropna().astype(str).unique().tolist())
    if not plans_for_band:
        st.error("Brak planów w wybranym paśmie.")
        st.stop()
    plan_sel = st.selectbox("Plan oceniany", plans_for_band, index=0)
with c3:
    # Domyślnie wszystkie plany w paśmie jako interferencja
    interferer_plans = st.multiselect(
        "Plany uwzględniane jako interferencja (w obrębie pasma)",
        plans_for_band,
        default=plans_for_band,
        help="Interferenci brani do analizy (nie tylko plan oceniany)."
    )

# Kanały wspólne Lo∩Hi dla planu ocenianego (do wyboru konkretnego kanału)
db_lo_plan = links_df[(links_df["band"]==band_sel) & (links_df["plan"]==plan_sel) & (links_df["chan_is_hi"]==False)]
db_hi_plan = links_df[(links_df["band"]==band_sel) & (links_df["plan"]==plan_sel) & (links_df["chan_is_hi"]==True)]
chans_common_ui = sorted(set(db_lo_plan["chan_num"].dropna().astype(int).tolist()) &
                         set(db_hi_plan["chan_num"].dropna().astype(int).tolist()))
if not chans_common_ui:
    chans_common_ui = sorted(set(db_lo_plan["chan_num"].dropna().astype(int).tolist()) |
                             set(db_hi_plan["chan_num"].dropna().astype(int).tolist()))
ch_sel = st.selectbox("Kanał (opcjonalnie)", ["(analizuj wszystkie)"] + [str(c) for c in chans_common_ui], index=0)

# Formularz: współrzędne + polaryzacja (NF i próg UKE ukryte – stałe)
with st.form("form_analiza"):
    st.subheader("Parametry łącza do oceny")
    cA, cB = st.columns(2)
    with cA:
        latA = st.text_input("LAT A", "52-27-37.05N")
        lonA = st.text_input("LON A", "16-52-52.59E")
    with cB:
        latB = st.text_input("LAT B", "52-13-45.00N")
        lonB = st.text_input("LON B", "21-00-41.00E")
    pol = st.selectbox("Polaryzacja", ["V","H"], index=0, help="Orientacja polaryzacji łącza kandydującego.")
    
    st.markdown("### Parametry anteny RX (dla HPBW)")
    rx_cols = st.columns([1, 1, 1])
    with rx_cols[0]:
        rx_hpbw_mode = st.radio("Sposób określenia HPBW RX", ["gain (dBi)", "średnica (m)"], index=0, horizontal=True)
    with rx_cols[1]:
        rx_gain_dbi_ui = st.number_input("Zysk RX [dBi]", min_value=20.0, max_value=60.0, value=46.0, step=0.5)
    with rx_cols[2]:
        rx_diam_m_ui = st.number_input("Średnica RX [m]", min_value=0.1, max_value=1.2, value=0.31, step=0.01, format="%.2f")

    submitted = st.form_submit_button("Analizuj", type="primary")

# =========================
# ANALIZA
# =========================
if submitted:
    # Koordy
    latA_dd = dms_tolerant_to_dd(latA); lonA_dd = dms_tolerant_to_dd(lonA)
    latB_dd = dms_tolerant_to_dd(latB); lonB_dd = dms_tolerant_to_dd(lonB)
    if any(np.isnan(x) for x in [latA_dd, lonA_dd, latB_dd, lonB_dd]):
        st.error("Nieprawidłowy format współrzędnych. Użyj np. 52-27-37.05N oraz 16-52-52.59E.")
        st.stop()

    # Parametry stałe (NF, próg UKE, promień)
    nf_db = DEFAULT_NF_DB
    radius_km = DEFAULT_RADIUS_BY_BAND_KM.get(int(band_sel), 30)

    dist_km = hav_km(latA_dd, lonA_dd, latB_dd, lonB_dd)
    az_ab = bearing_deg(latA_dd, lonA_dd, latB_dd, lonB_dd)
    az_ba = bearing_deg(latB_dd, lonB_dd, latA_dd, lonA_dd)
    st.info(
        f"Dystans A–B ≈ **{dist_km:.2f} km** • "
        f"Azymut A→B: **{az_ab:.1f}°** • Azymut B→A: **{az_ba:.1f}°** • "
        f"Promień analizy (auto): **{radius_km:.0f} km**"
    )

    # Baza interferentów Lo/Hi – uwzględnia WSZYSTKIE wybrane plany w PASMIE
    db_lo_all = links_df[(links_df["band"]==band_sel) & (links_df["chan_is_hi"]==False) & (links_df["plan"].isin(interferer_plans))]
    db_hi_all = links_df[(links_df["band"]==band_sel) & (links_df["chan_is_hi"]==True)  & (links_df["plan"].isin(interferer_plans))]

    # Prefiltr promienia względem RX w danym kierunku
    db_lo = filter_by_radius(db_lo_all, latB_dd, lonB_dd, radius_km)  # A→B: Rx = B
    db_hi = filter_by_radius(db_hi_all, latA_dd, lonA_dd, radius_km)  # B→A: Rx = A

    if db_lo.empty and db_hi.empty:
        st.error("Po prefiltrze promienia brak interferentów. Zwiększ listę planów lub wybierz inne położenie.")
        st.stop()

    # Szerokość kanału planu ocenianego (mediana), do I/N i widma
    bw_mhz_plan = float(pd.to_numeric(
        links_df.loc[(links_df["band"]==band_sel) & (links_df["plan"]==plan_sel), "bw_mhz"],
        errors="coerce"
    ).dropna().median()) if not links_df.empty else 250.0
    if pd.isna(bw_mhz_plan) or bw_mhz_plan <= 0:
        bw_mhz_plan = 250.0

    # Reprezentatywne f (fallback – właściwe f_c liczymy per kanał z lookupu)
    f_lo_ghz = float(db_lo["f_ghz"].dropna().astype(float).mean()) if not db_lo.empty else float(band_sel)
    f_hi_ghz = float(db_hi["f_ghz"].dropna().astype(float).mean()) if not db_hi.empty else float(band_sel)

    # Kandydat
    cand = {
        "latA": latA_dd, "lonA": lonA_dd,
        "latB": latB_dd, "lonB": lonB_dd,
        "pol": pol, "band": int(band_sel)
    }
    cand["rx_hpbw_mode"] = "gain" if rx_hpbw_mode.startswith("gain") else "diameter"
    cand["rx_gain_dbi"] = rx_gain_dbi_ui
    cand["rx_diam_m"] = rx_diam_m_ui
    # Kanał narzucony z UI?
    plan_channels = None
    if ch_sel != "(analizuj wszystkie)":
        plan_channels = [int(ch_sel)]

    # Lookup częstotliwości środkowych kanałów
    fc_lookup = build_fc_lookup(links_df)

    # Dwukierunkowo
    res = score_bidir(cand, db_lo, db_hi, f_lo_ghz, f_hi_ghz,
                      bw_mhz_plan=bw_mhz_plan, plan_sel=plan_sel,
                      fc_lookup=fc_lookup, nf_db_ui=nf_db,
                      plan_channels=plan_channels)
    if res.empty:
        st.error("Brak kanałów do oceny (po filtrach).")
        st.stop()

# --- Bezpieczne inicjalizacje (na wypadek braku rekomendacji) ---
    best = None
    best_channel = None
    uke_ok = False

    # --- Rekomendacja kanału z kryterium UKE ---
    # --- Rekomendacja kanału z kryterium UKE (twardy filtr) ---
    eligible = res[res["UKE_ok"]].copy()

    if not eligible.empty:
        # sortuj bardziej "UKE-owo": minimalna degradacja, potem I/N, a na końcu kanał (stabilny tie-break)
        best = (eligible
                .sort_values(["Degr_max_dB", "IN_max_dB", "channel"], kind="mergesort")
                .iloc[0])
        best_channel = int(best["channel"])
        compliance_note = "✅ spełnia UKE (degradacja ≤ 1 dB w obu kierunkach)"
        uke_ok = True
        txt_band = "E‑Band 70/80 GHz" if band_sel == 80 else f"{band_sel} GHz"
        best_fwd = float(best.get("IN_fwd_dB", np.nan))
        best_rev = float(best.get("IN_rev_dB", np.nan))

        st.success(
            f"**Rekomendacja (dwukierunkowo):** kanał **{best_channel}** • "
            f"Plan: **{plan_sel}**, Pasmo: **{txt_band}**  \n"
            f"**I/N max = {best['IN_max_dB']:.1f} dB** "
            f"(fwd={best_fwd:.1f} dB, rev={best_rev:.1f} dB) • "
            f"**Degradacja max = {best['Degr_max_dB']:.2f} dB** • {compliance_note}  \n"
            f"Szum N≈**{best['N_dbm']:.1f} dBm** dla BW≈**{bw_mhz_plan:.0f} MHz**, NF≈**{nf_db:.1f} dB**"
        )

        # Dodatkowa linia ze szczegółami I/N i degradacji per kierunek
        st.success(
            f"I/N (fwd/rev): {best['IN_fwd_dB']:.1f}/{best['IN_rev_dB']:.1f} dB • "
            f"Degradacja (fwd/rev): {best['Degr_fwd_dB']:.2f}/{best['Degr_rev_dB']:.2f} dB"
        )

    else:
        # Brak kanałów spełniających UKE => nie rekomendujemy nic
        st.warning(
            "Brak kanału spełniającego UKE (Degradacja_max ≤ 1 dB). "
            "Poniżej 3 najbliższe progu 1 dB (to **nie** jest rekomendacja)."
        )
        near = res.sort_values(["Degr_max_dB", "IN_max_dB", "channel"], kind="mergesort").head(3)
        cols_show = [
            "channel", "IN_fwd_dB", "IN_rev_dB", "IN_max_dB",
            "Degr_fwd_dB", "Degr_rev_dB", "Degr_max_dB",
            "I_fwd_dBm", "I_rev_dBm", "N_dbm", "UKE_ok"
        ]
        for c in ["I_fwd_dBm", "I_rev_dBm"]:
            if c not in near.columns:
                near[c] = np.nan
        st.dataframe(near[cols_show], use_container_width=True, height=240)
        st.stop()

    # --- Wykres UKE: Degradacja (słupki, oś lewa) + I/N fwd/rev (linie, oś prawa) ---
    plot_df = res.copy().sort_values("channel")  # naturalny porządek kanałów
    bar_colors = np.where(plot_df["UKE_ok"], "seagreen", "crimson")

    fig = go.Figure()

    # 1) Degradacja max [dB] – słupki (oś lewa, priorytet UKE)
    fig.add_trace(go.Bar(
        x=plot_df["channel"],
        y=plot_df["Degr_max_dB"],
        name="Degradacja max [dB] (kryterium UKE)",
        marker_color=bar_colors,
        yaxis="y1",
        text=[f"{v:.2f}" if pd.notna(v) else "" for v in plot_df["Degr_max_dB"]],
        textposition="outside",
        cliponaxis=False
    ))

    # 2) I/N fwd i rev [dB] – linie (oś prawa)
    fig.add_trace(go.Scatter(
        x=plot_df["channel"],
        y=plot_df["IN_fwd_dB"],
        name="I/N fwd [dB]",
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=8),
        yaxis="y2",
        hovertemplate="Kanał %{x}<br>I/N fwd: %{y:.1f} dB<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["channel"],
        y=plot_df["IN_rev_dB"],
        name="I/N rev [dB]",
        mode="lines+markers",
        line=dict(color="#ff7f0e"),
        marker=dict(size=8),
        yaxis="y2",
        hovertemplate="Kanał %{x}<br>I/N rev: %{y:.1f} dB<extra></extra>"
    ))

    # 3) Linia progu UKE = 1 dB (na osi degradacji)
    fig.add_shape(
        type="line", xref="paper", yref="y1",
        x0=0, x1=1, y0=1.0, y1=1.0,
        line=dict(color="orange", width=2, dash="dash")
    )
    fig.add_annotation(
        x=1.0, y=1.0, xref="paper", yref="y1",
        text="Próg UKE 1 dB", showarrow=False, xanchor="right",
        font=dict(color="orange")
    )

    # 4) Znacznik rekomendowanego kanału (na degradacji)
    try:
        y_star = float(plot_df.loc[plot_df["channel"] == best_channel, "Degr_max_dB"].iloc[0])
        fig.add_trace(go.Scatter(
            x=[best_channel], y=[y_star],
            mode="markers+text",
            text=["★"], textposition="top center",
            marker=dict(color="gold", size=16, symbol="star"),
            name="Rekomendacja (min Degr_max)"
        ))
    except Exception:
        pass

    fig.update_layout(
        title="Kanały – Degradacja (słupki, oś lewa) + I/N fwd/rev (linie, oś prawa). Zielone = UKE OK (≤ 1 dB).",
        xaxis=dict(title="Kanał"),
        yaxis=dict(title="Degradacja [dB] (UKE ≤ 1 dB)"),                # oś lewa
        yaxis2=dict(title="I/N [dB] (mniejsze = lepiej)", overlaying="y", side="right"),  # oś prawa
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.25,
        margin=dict(l=60, r=60, t=70, b=60),
        hovermode="x unified",
        uniformtext_minsize=10, uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Tabela + eksport CSV ---
    cols_show = [
        "channel",
        "IN_fwd_dB","IN_rev_dB","IN_max_dB",
        "Degr_fwd_dB","Degr_rev_dB","Degr_max_dB",
        "I_fwd_dBm","I_rev_dBm","N_dbm",
        "UKE_ok"
    ]
    for c in ["I_fwd_dBm","I_rev_dBm"]:
        if c not in res.columns:
            res[c] = np.nan
    st.dataframe(res[cols_show], use_container_width=True, height=480)
    st.download_button(
        "Pobierz ranking (CSV, z degradacją i UKE_ok)",
        res[cols_show].to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_{band_sel}GHz_{plan_sel}_BIDIR_IN_DEGR.csv",
        mime="text/csv"
    )

    
    # === Top interferenci dla gorszego kierunku (mapa + tabela) ===
    worst_dir = "fwd" if best["IN_fwd_dB"] >= best["IN_rev_dB"] else "rev"

    if worst_dir == "fwd":
        db_used = db_lo
        rx_lat, rx_lon = latB_dd, lonB_dd
        label_dir = "A→B (Lo)"
        # f_c z lookupu dla best_channel
        f_c_best = get_fc(build_fc_lookup(links_df), band_sel, plan_sel, False, best_channel, f_lo_ghz)
        # kierunek RX: B -> A
        rx_brg_dir = bearing_deg(rx_lat, rx_lon, latA_dd, lonA_dd)
    else:
        db_used = db_hi
        rx_lat, rx_lon = latA_dd, lonA_dd
        label_dir = "B→A (Hi)"
        f_c_best = get_fc(build_fc_lookup(links_df), band_sel, plan_sel, True, best_channel, f_hi_ghz)
        # kierunek RX: A -> B
        rx_brg_dir = bearing_deg(rx_lat, rx_lon, latB_dd, lonB_dd)

    # --- HPBW RX (odbiornika kandydata) wg trybu z UI (gain lub średnica)
    if rx_hpbw_mode.startswith("gain"):
        rx_hpbw_best = hpbw_from_gain(
            np.array([rx_gain_dbi_ui]),
            np.array([f_c_best]),
            fallback_d_m=(0.3 if int(band_sel) == 80 else DEFAULT_DIAM_M)
        )[0]
    else:
        rx_hpbw_best = hpbw_deg(f_c_best, rx_diam_m_ui)

    if not db_used.empty:
        # wektory interferentów
        txi = db_used.copy()
        txi_lat = txi["tx_lat_dd"].to_numpy()
        txi_lon = txi["tx_lon_dd"].to_numpy()
        eirp_i  = txi["eirp_dbm"].to_numpy()
        pol_i   = txi["pol"].astype(str).str.upper().str[0].to_numpy()
        f_i     = txi["f_ghz"].to_numpy()
        bw_i = pd.to_numeric(
            txi.get("bw_mhz", pd.Series(np.nan, index=txi.index)), errors="coerce"
        ).fillna(
            pd.to_numeric(txi["bw_mhz"], errors="coerce").median() if "bw_mhz" in txi.columns else bw_mhz_plan
        ).to_numpy()
        tx_brg_i = txi["tx_bearing"].to_numpy()

        # Geometria
        D    = hav_km(txi_lat, txi_lon, np.full_like(txi_lat, rx_lat), np.full_like(txi_lon, rx_lon))
        be_tx = bearing_deg(txi_lat, txi_lon, np.full_like(txi_lat, rx_lat), np.full_like(txi_lon, rx_lon))
        be_rx = bearing_deg(np.full_like(txi_lat, rx_lat), np.full_like(txi_lon, rx_lon), txi_lat, txi_lon)
        ang_tx = np.abs((tx_brg_i - be_tx + 180) % 360 - 180)
        ang_rx = np.abs((rx_brg_dir - be_rx + 180) % 360 - 180)

        # --- HPBW TX interferentów z ich zysku anteny (fallback: 0.3 m w E-band, inaczej DEFAULT_DIAM_M)
        tx_gain_i = pd.to_numeric(txi.get("tx_gain_dbi", pd.Series(np.nan, index=txi.index)), errors="coerce").to_numpy()
        tx_hpbw_i = hpbw_from_gain(
            tx_gain_i, f_i,
            fallback_d_m=(0.3 if int(band_sel) == 80 else DEFAULT_DIAM_M)
        )

        FSPL    = np.array([fspl_db(d, fi) for d, fi in zip(D, f_i)])
        XPD_arr = np.where(pol_i == pol, 0.0, XPD_DB)
        
        
        # --- PSD + integracja widma + ACIR = MIN(ACIR, Overlap) (TOP lista) ---

        # 1) Parametry pasm
        B_rx_Hz = float(bw_mhz_plan) * 1e6
        df_mhz  = (f_i - f_c_best) * 1000.0

        # Overlap (MHz) -> Hz
        ov_mhz  = np.maximum(0.0, bw_i/2.0 + bw_mhz_plan/2.0 - np.abs(df_mhz))
        B_ov_Hz = ov_mhz * 1e6

        # 2) ACI: Overlap + ACIR(Δf) i konserwatywne MIN
        Adj_overlap = spectral_adj_db_vec(f_i, bw_i, f_c_best, bw_mhz_plan)
        Adj_acir    = acir_step_db(df_mhz, th_125=45.0, th_250=55.0, th_375=60.0, th_500=65.0, th_ge=70.0)
        Adj         = aci_combine_min(Adj_acir, Adj_overlap)

        # 3) PSD interferenta (dBm/Hz) i integracja po BW
        eirp_dens_dBm_per_Hz = eirp_i - 10.0*np.log10(np.maximum(bw_i, 1e-6) * 1e6)
        B_int_Hz = np.where(B_ov_Hz > 0.0, B_ov_Hz, B_rx_Hz)

        # 4) Poziom wkładu interferenta (dBm)
        Pint = (eirp_dens_dBm_per_Hz
                - Adj
                + 10.0*np.log10(B_int_Hz)            # integracja PSD po B_int
                - FSPL
                - offaxis_parabolic(ang_tx, tx_hpbw_i)
                - offaxis_parabolic(ang_rx, rx_hpbw_best)
                - XPD_arr)

        # 5) Quasi‑cochannel penalty (jak w rankingu), UWAGA: teraz w Hz
        min_bw   = np.minimum(bw_i, bw_mhz_plan)
        quasi_co = B_ov_Hz >= (0.8 * min_bw * 1e6)
        aligned  = (ang_tx < 3 * tx_hpbw_i) & (ang_rx < 3 * rx_hpbw_best)
        Pint[(D < reuse_for_band(band_sel)) & aligned & quasi_co] += 100.0


        # Tabela + mapa
        txi_map = txi[["tx_lon", "tx_lat", "plan", "chan_num"]].copy()
        txi_map["tx_lon"] = pd.to_numeric(txi_map["tx_lon"], errors="coerce")
        txi_map["tx_lat"] = pd.to_numeric(txi_map["tx_lat"], errors="coerce")
        txi_map["Pint_dBm"] = Pint
        txi_map["rx_lon"] = rx_lon
        txi_map["rx_lat"] = rx_lat
        txi_map = txi_map.sort_values("Pint_dBm", ascending=False).head(30)

        st.subheader(f"Największe wkłady interferencyjne – kanał {best_channel}, kierunek {label_dir}")
        st.caption(f"Azymut RX w tym kierunku: {rx_brg_dir:.1f}°")
        st.dataframe(txi_map[["plan", "chan_num", "tx_lon", "tx_lat", "Pint_dBm"]],
                    use_container_width=True, height=360)

        st.subheader("Mapa (pydeck)")
        

        st.subheader("Mapa (pydeck)")
        deck = make_map(latA_dd, lonA_dd, latB_dd, lonB_dd, txi_map, label_dir, radius_km)
        st.pydeck_chart(deck, use_container_width=True)