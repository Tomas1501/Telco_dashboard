# 08_LOS_profile_fixed.py
# -*- coding: utf-8 -*-
"""
Profil NMT/NMPT + LoS + Fresnel
- Minimalny UI (Streamlit + Plotly)
- Domyślne anteny: 50 m nad NMT (offset)
- Krzywizna Ziemi: k = 0.13 (włącz/wyłącz)
- F1-clearance = 0.80 (80% F1)
- Raport przeszkód: segmenty z maks. penetracją ponad LoS i ponad 0.8·F1
- Globalne statusy: NO-LOS (jeśli gdziekolwiek przecina LoS), "80% F1 wolne" (jeśli wszędzie spełnione)
- Eksport wyłącznie PNG (Plotly -> kaleido)
"""

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


hide_toolbar = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_toolbar, unsafe_allow_html=True)

st.set_page_config(page_title="Profil NMT/NMPT + LoS + Fresnel", layout="wide")

# -------------------- Parsowanie TXT Geoportal (X, Y, Z) --------------------
FLOAT_RE = r"[\-+]?(?:\d+(?:[\.,]\d*)?|[\.,]\d+)(?:[eE][\-+]?\d+)?"

def _to_float(s: str) -> float:
    return float(str(s).replace(",", "."))

def parse_geoportal_txt(content: bytes) -> pd.DataFrame:
    """
    Czyta TXT/CSV Geoportalu z X,Y,Z (ignoruje nagłówki/segmenty) i liczy dystans skumulowany (Euklides).
    """
    text = None
    for enc in ("utf-8", "cp1250", "iso-8859-2", "latin-2"):
        try:
            text = content.decode(enc); break
        except Exception:
            continue
    if text is None:
        text = content.decode("utf-8", errors="ignore")

    triple = re.compile(rf"\s*({FLOAT_RE})\s*[,;\t ]\s*({FLOAT_RE})\s*[,;\t ]\s*({FLOAT_RE})\s*$")
    X, Y, Z = [], [], []
    for line in text.splitlines():
        m = triple.match(line.strip())
        if m:
            X.append(_to_float(m.group(1)))
            Y.append(_to_float(m.group(2)))
            Z.append(_to_float(m.group(3)))

    if len(X) < 2:
        raise RuntimeError("Plik TXT/CSV ma < 2 punktów X,Y,Z.")

    X = np.array(X, float); Y = np.array(Y, float); Z = np.array(Z, float)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z, "distance_m": d})

# -------------------- Wyrównanie NMPT do siatki NMT --------------------
def align_nmpt_to_nmt(nmt_df: pd.DataFrame, nmpt_df: pd.DataFrame | None):
    """
    Zwraca: dist (NMT), ground=NMT(Z), surface=NMPT w siatce NMT (jeśli jest).
    Gdy NMPT brak -> surface=NMT. Dodatkowo porządkuje i deduplikuje dystanse.
    """
    d_nmt = nmt_df["distance_m"].to_numpy(dtype=float)
    z_nmt = nmt_df["Z"].to_numpy(dtype=float)
    order_nmt = np.argsort(d_nmt)
    d_nmt = d_nmt[order_nmt]; z_nmt = z_nmt[order_nmt]
    keep_nmt = np.append([True], d_nmt[1:] > d_nmt[:-1])
    d_nmt = d_nmt[keep_nmt]; z_nmt = z_nmt[keep_nmt]

    if nmpt_df is None:
        return d_nmt, z_nmt, z_nmt.copy(), False

    d_nmpt = nmpt_df["distance_m"].to_numpy(dtype=float)
    z_nmpt = nmpt_df["Z"].to_numpy(dtype=float)
    order_sm = np.argsort(d_nmpt)
    d_nmpt = d_nmpt[order_sm]; z_nmpt = z_nmpt[order_sm]
    keep_sm = np.append([True], d_nmpt[1:] > d_nmpt[:-1])
    d_nmpt = d_nmpt[keep_sm]; z_nmpt = z_nmpt[keep_sm]

    z_nmpt_on_nmt = np.interp(d_nmt, d_nmpt, z_nmpt, left=z_nmpt[0], right=z_nmpt[-1])
    return d_nmt, z_nmt, z_nmpt_on_nmt, True

# -------------------- Obliczenia LoS + Fresnel --------------------
def compute_profile(dist, ground_asl, surface_asl,
                    freq_ghz=23.0,
                    use_curvature=True, refr_k=0.13,
                    fresnel_clearance=0.80,
                    start_mode="offset", start_off=50.0, start_asl=120.0,   # <-- domyślnie 50 m
                    end_mode="offset", end_off=50.0, end_asl=120.0):        # <-- domyślnie 50 m
    """
    Zwraca:
      df (profil z kolumnami: distance_m, ground, surface, los, fresnel_r1_m, los_minus_kF1, los_minus_F1, state)
      stats (L, lambda, r1_mid)
    """
    dist = np.asarray(dist, float)
    ground = np.asarray(ground_asl, float)
    surface = np.asarray(surface_asl, float)

    # unifikacja startu na zero + porządkowanie
    dist = dist - dist.min()
    order = np.argsort(dist)
    dist = dist[order]; ground = ground[order]; surface = surface[order]
    keep = np.append([True], dist[1:] > dist[:-1])
    dist = dist[keep]; ground = ground[keep]; surface = surface[keep]

    L = dist.max()
    if L <= 0:
        raise RuntimeError("Długość profilu = 0.")

    # krzywizna (k=0.13): uginamy powierzchnie
    if use_curvature:
        R = 6_371_000.0
        Reff = R / (1.0 - refr_k) if 0 <= refr_k < 1.0 else R
        corr = (dist**2) / (2.0 * Reff)
        ground = ground + corr
        surface = surface + corr

    # anteny: tryb offset od NMT (ground)
    z_start = (start_asl if start_mode == "absolute" else ground[0] + start_off)
    z_end   = (end_asl   if end_mode   == "absolute" else ground[-1] + end_off)

    # LoS (prosta)
    los = z_start + (z_end - z_start) * (dist / L)

    # Fresnel (1. strefa)
    c = 299_792_458.0
    lam = c / (freq_ghz * 1e9)
    r1 = np.sqrt(np.maximum(0.0, lam * dist * (L - dist) / L))
    los_minus_kF1 = los - fresnel_clearance * r1
    los_minus_F1  = los - r1

    # klasyfikacja punktowa (zostawiona tylko dla markerów na wykresie)
    tol = 1e-2  # 1 cm
    is_blocked = surface > (los + tol)
    is_intrus  = (~is_blocked) & (surface > (los_minus_kF1 + tol))
    state = np.where(is_blocked, "LOS_BLOCKED",
             np.where(is_intrus, "FRESNEL_INTRUSION", "CLEAR"))

    df = pd.DataFrame({
        "distance_m": dist,
        "ground_m_asl": ground,
        "surface_m_asl": surface,
        "los_m_asl": los,
        "fresnel_r1_m": r1,
        "los_minus_kF1_m_asl": los_minus_kF1,
        "los_minus_F1_m_asl": los_minus_F1,
        "state": state
    })

    stats = {
        "L": float(L),
        "lambda_m": float(lam),
        "r1_mid_m": float(r1[np.argmin(np.abs(dist - 0.5*L))])
    }
    return df, stats

# -------------------- Detekcja przeszkód (segmenty) --------------------
def _find_segments(mask: np.ndarray):
    """Zwraca listę (i0, i1) dla kolejnych spójnych fragmentów, gdzie mask==True."""
    segs = []
    if mask.size == 0:
        return segs
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True; start = i
        elif not v and in_run:
            in_run = False; segs.append((start, i-1))
    if in_run:
        segs.append((start, mask.size-1))
    return segs

def extract_obstacles(dist: np.ndarray,
                      surface: np.ndarray,
                      los: np.ndarray,
                      r1: np.ndarray,
                      k_clear: float = 0.80,
                      tol: float = 1e-2):
    """
    Identyfikuje przeszkody:
      - LOS_BLOCKER: surface > LoS + tol  (blokuje wiązkę)
      - F1_INTRUDER: surface > (LoS - k*r1) + tol, ale nie przekracza LoS (narusza 80% F1)
    Zwraca:
      df_obs (tabela przeszkód),
      global_los_ok (True jeśli brak blokady LoS),
      global_kF1_ok (True jeśli wszędzie spełniony warunek k*F1 wolne),
      worst_kF1_margin_m (min(LoS - k*r1 - surface))  # <0 oznacza naruszenie
    """
    dist = np.asarray(dist, float)
    surface = np.asarray(surface, float)
    los = np.asarray(los, float)
    r1 = np.asarray(r1, float)

    cl = los - k_clear * r1
    los_block = surface > (los + tol)
    f1_intr   = (~los_block) & (surface > (cl + tol))

    seg_los = _find_segments(los_block)
    seg_f1  = _find_segments(f1_intr)

    rows = []
    for (i0, i1) in seg_los:
        d0, d1 = float(dist[i0]), float(dist[i1])
        sl = slice(i0, i1+1)
        pen_los = float(np.max(surface[sl] - los[sl]))
        pen_kf1 = float(np.max(surface[sl] - cl[sl]))
        rows.append({
            "typ": "LOS_BLOCKER",
            "d_start_m": d0,
            "d_end_m": d1,
            "len_m": d1 - d0,
            "max_pen_over_LoS_m": round(pen_los, 2),
            "max_pen_over_kF1_m": round(pen_kf1, 2),
        })

    for (i0, i1) in seg_f1:
        d0, d1 = float(dist[i0]), float(dist[i1])
        sl = slice(i0, i1+1)
        pen_los = float(np.max(surface[sl] - los[sl]))
        pen_kf1 = float(np.max(surface[sl] - cl[sl]))
        rows.append({
            "typ": "F1_INTRUDER",
            "d_start_m": d0,
            "d_end_m": d1,
            "len_m": d1 - d0,
            "max_pen_over_LoS_m": round(pen_los, 2),   # zwykle ≤ 0 (+tol)
            "max_pen_over_kF1_m": round(pen_kf1, 2),   # > 0
        })

    columns = ["typ", "d_start_m", "d_end_m", "len_m", "max_pen_over_LoS_m", "max_pen_over_kF1_m"]
    if rows:
        df_obs = pd.DataFrame(rows, columns=columns).sort_values(["typ", "d_start_m"], kind="mergesort").reset_index(drop=True)
    else:
        # Pusta ramka, ale z kolumnami — żeby nie było KeyError: 'typ' przy sortowaniu / wyświetlaniu
        df_obs = pd.DataFrame(columns=columns)

    global_los_ok = (len(seg_los) == 0)
    worst_kF1_margin = float(np.min(cl - surface)) if surface.size else 0.0
    global_kF1_ok = (worst_kF1_margin >= -tol)

    return df_obs, global_los_ok, global_kF1_ok, worst_kF1_margin

# -------------------- Wykres (Plotly) --------------------
def plot_profile(df: pd.DataFrame, show_full_f1=False):
    rename = {
        "ground_m_asl": "NMT (grunt)",
        "surface_m_asl": "NMPT (pokrycie)",
        "los_m_asl": "Linia LoS",
        "los_minus_kF1_m_asl": "k·F1 od LoS"
    }
    cols = ["ground_m_asl", "surface_m_asl", "los_m_asl", "los_minus_kF1_m_asl"]
    if show_full_f1:
        rename["los_minus_F1_m_asl"] = "100% F1 od LoS"
        cols.append("los_minus_F1_m_asl")

    lines = (df[["distance_m"] + cols]
             .rename(columns=rename)
             .melt(id_vars="distance_m", var_name="seria", value_name="wys"))

    fig = px.line(
        lines, x="distance_m", y="wys", color="seria",
        color_discrete_map={
            "NMT (grunt)": "#666666",
            "NMPT (pokrycie)": "#22BB77",
            "Linia LoS": "red",
            "k·F1 od LoS": "royalblue",
            "100% F1 od LoS": "steelblue",
        },
        template="plotly_white"
    )

    for tr in fig.data:
        if tr.name == "k·F1 od LoS":
            tr.line.dash = "dash"; tr.line.width = 1.0
        elif tr.name == "100% F1 od LoS":
            tr.line.dash = "dot"; tr.line.width = 0.8
        elif tr.name == "Linia LoS":
            tr.line.width = 1.4
        elif tr.name == "NMT (grunt)":
            tr.line.width = 1.6
        elif tr.name == "NMPT (pokrycie)":
            tr.line.width = 1.2

    # markery wg stanu (po "surface")
    names = {
        "CLEAR": "OK (LoS i Fresnel)",
        "FRESNEL_INTRUSION": "Naruszenie Fresnela (LoS wolny)",
        "LOS_BLOCKED": "LoS zablokowany"
    }
    colors = {"CLEAR": "green", "FRESNEL_INTRUSION": "orange", "LOS_BLOCKED": "red"}
    for s in ["CLEAR", "FRESNEL_INTRUSION", "LOS_BLOCKED"]:
        sub = df[df["state"] == s]
        if not sub.empty:
            fig.add_scatter(x=sub["distance_m"], y=sub["surface_m_asl"],
                            mode="markers", name=names[s],
                            marker=dict(size=6, color=colors[s]), showlegend=True)

    fig.update_layout(
        title="Profil NMT/NMPT + LoS + Fresnel",
        xaxis_title="Dystans [m]", yaxis_title="Wysokość [m n.p.m.]",
        legend=dict(orientation="h", y=1.02, x=0),
        height=420, margin=dict(l=40, r=10, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    return fig

# -------------------- UI --------------------
st.title("Profil NMT/NMPT + LoS + Fresnel (TXT z Geoportalu)")
st.subheader('https://mapy.geoportal.gov.pl/imap/Imgp_2.html?gpmap=gp0 - wyrysuj linię (w profil terenu)')
st.caption("Pobierz pliki NMT oraz NMPT txt  i  wgraj poniżej")

col1, col2 = st.columns(2)
with col1:
    nmt_file = st.file_uploader("**NMT (grunt)** — TXT Geoportalu", type=["txt", "csv"])
with col2:
    nmpt_file = st.file_uploader("**NMPT (pokrycie)** — TXT Geoportalu", type=["txt", "csv"])

if nmt_file:
    try:
        nmt_df = parse_geoportal_txt(nmt_file.read())
        nmpt_df = parse_geoportal_txt(nmpt_file.read()) if nmpt_file else None
        dist, ground, nmpt_on_nmt, has_nmpt = align_nmpt_to_nmt(nmt_df, nmpt_df)

        st.subheader("Parametry")
        c1, c2, c3 = st.columns(3)
        with c1:
            freq_ghz = st.number_input("Częstotliwość [GHz]", min_value=0.1, max_value=120.0, value=23.0, step=0.1)
            use_curv = st.checkbox("Korekta krzywizny Ziemi (k = 0.13)", value=True)
            show_full_f1 = st.checkbox("Pokaż także 100% F1", value=False)
        with c2:
            start_mode = st.radio("Wysokość początku", ["offset", "absolute"], index=0)
            start_off = st.number_input("Offset nad NMT (początek) [m]", value=50.0, step=0.1, disabled=(start_mode=="absolute"))  # 50 m
            start_asl = st.number_input("ASL (początek) [m]", value=120.0, step=0.1, disabled=(start_mode=="offset"))
        with c3:
            end_mode = st.radio("Wysokość końca", ["offset", "absolute"], index=0)
            end_off = st.number_input("Offset nad NMT (koniec) [m]", value=50.0, step=0.1, disabled=(end_mode=="absolute"))      # 50 m
            end_asl = st.number_input("ASL (koniec) [m]", value=120.0, step=0.1, disabled=(end_mode=="offset"))

        use_nmpt = st.checkbox("Uwzględnij NMPT (pokrycie) w analizie", value=has_nmpt, disabled=not has_nmpt)
        surface_for_calc = nmpt_on_nmt if (has_nmpt and use_nmpt) else ground
        st.info(f"Analiza kolizji względem: **{'NMPT (pokrycie)' if (has_nmpt and use_nmpt) else 'NMT (grunt)'}**")

        df_res, stats = compute_profile(
            dist, ground, surface_for_calc,
            freq_ghz=freq_ghz,
            use_curvature=use_curv, refr_k=0.13,
            fresnel_clearance=0.80,
            start_mode=start_mode, start_off=float(start_off or 0.0), start_asl=float(start_asl or 0.0),
            end_mode=end_mode, end_off=float(end_off or 0.0), end_asl=float(end_asl or 0.0)
        )

        # --- Raport przeszkód i globalne statusy ---
        df_obs, los_ok, kF1_ok, worst_margin = extract_obstacles(
            dist=df_res["distance_m"].to_numpy(),
            surface=df_res["surface_m_asl"].to_numpy(),
            los=df_res["los_m_asl"].to_numpy(),
            r1=df_res["fresnel_r1_m"].to_numpy(),
            k_clear=0.80, tol=1e-2
        )

        cG1, cG2, cG3 = st.columns(3)
        cG1.metric("Status LoS", "OK" if los_ok else "NO-LOS")
        cG2.metric("Warunek 80% F1 wolne", "Spełniony" if kF1_ok else "NIESPEŁNIONY")
        cG3.metric("Najgorszy margines do 0.8·F1 [m]", f"{worst_margin:.2f}")

        

        # Wykres
        fig = plot_profile(df_res, show_full_f1=show_full_f1)
        # Jeśli analizujesz „sam grunt”, a masz NMPT – pokazujemy podgląd NMPT (nie wpływa na analizę)
        if has_nmpt and not use_nmpt:
            fig.add_scatter(x=dist, y=nmpt_on_nmt, name="NMPT (pokrycie) — podgląd (nieużyte)",
                            line=dict(color="#22BB77", width=1, dash="dot"), opacity=0.6, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        # TYLKO długość (usunięto % udziałów)
        st.metric("Długość trasy [m]", f"{stats['L']:.0f}")

        # Debug Fresnela (zostawiam)
        with st.expander("Debug Fresnela", expanded=False):
            st.markdown(
                f"- Długość fali λ = **{stats['lambda_m']:.3e} m**  \n"
                f"- r₁(L/2) = **{stats['r1_mid_m']:.2f} m**  \n"
                f"- k·F1(L/2) (k=0.80) = **{0.80*stats['r1_mid_m']:.2f} m**  \n"
                f"- Długość trasy L = **{stats['L']:.1f} m**"
            )
        with st.expander("Przeszkody i penetracja (wysokościowo)", expanded=not los_ok or not kF1_ok):
            if df_obs.empty:
                st.write("Brak wykrytych przeszkód (ani LoS, ani 0.8·F1).")
            else:
                st.dataframe(df_obs, use_container_width=True)
            st.caption("Jeśli **jakakolwiek** przeszkoda przecina LoS → globalny status **NO‑LOS**. "
                       "Warunek 80% F1: wszędzie surface ≤ LoS − 0.8·F1 (margines ≥ 0).")
        st.subheader("Eksport")
        try:
            png = fig.to_image(format="png", scale=2)  # wymaga: pip install -U kaleido
            st.download_button("Pobierz wykres PNG", data=png, file_name="profil_fresnel.png", mime="image/png")
        except Exception:
            st.info("Aby eksportować PNG, zainstaluj **kaleido**: `pip install -U kaleido`.")

    except Exception as e:
        st.error(f"Błąd: {e}")
else:
    st.info("Wgraj **NMT.txt** (wymagane) i opcjonalnie **NMPT.txt**. Oba w formacie X,Y,Z z Geoportalu.")
