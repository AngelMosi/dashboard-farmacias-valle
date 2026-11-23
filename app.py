from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Farmacias del Valle | Vista Ejecutiva",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Sin cabecera de Streamlit (evita el título cortado y el botón Deploy) y espaciado compacto.
st.markdown(
    """
    <style>
        header[data-testid="stHeader"], #MainMenu, footer {display: none;}
        .block-container {padding-top: 0.9rem; padding-bottom: 0.2rem; max-width: 98%;}
        div[data-testid="stMetric"] {
            padding: 0.4rem 0.7rem;
            border: 1px solid rgba(150, 160, 180, 0.25);
            border-radius: 8px;
        }
        div[data-testid="stMetricValue"] {font-size: 1.3rem; font-weight: 700;}
        div[data-testid="stMetricDelta"] {font-size: 0.78rem;}
        hr {margin: 0.4rem 0;}
        .titulo {font-size: 1.3rem; font-weight: 700; line-height: 1.25; margin: 0 0 0.15rem 0;}
        .sub {font-size: 0.8rem; opacity: 0.75; margin-bottom: 0.4rem;}
        .pie {font-size: 0.72rem; opacity: 0.7; margin-top: 0.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).resolve().parent
DATA_FILE = next(iter(sorted(BASE.glob("Farmacias_Valle_Demanda*.csv"))), None)
METAS_FILE = BASE / "metas_anuales.csv"

# Paleta mínima con significado: azul = énfasis, ámbar = alerta, gris = contexto.
AZUL, AMBAR, GRIS = "#38bdf8", "#f59e0b", "#64748b"


@st.cache_data
def cargar_datos(csv_path: Path, metas_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if metas_path.exists():
        metas = pd.read_csv(metas_path)
    else:  # respaldo: misma meta constante de cobertura si no está metas_anuales.csv
        metas = pd.DataFrame({"Anio": [2020, 2021, 2022, 2023, 2024], "Cobertura_Meta_Meses": [1.60] * 5})
    df["Fecha"] = pd.to_datetime(df["Mes"] + "-01")
    df["Anio"] = df["Fecha"].dt.year
    df["Cobertura_Meses"] = np.where(
        df["Unidades_Vendidas"] > 0, df["Stock_Disponible"] / df["Unidades_Vendidas"], np.nan
    )
    df = df.merge(metas, on="Anio", how="left")
    df["Bajo_Meta"] = df["Cobertura_Meses"] < df["Cobertura_Meta_Meses"]
    # Unidades relativas al promedio de cada producto: aísla el efecto de la promoción
    # del tamaño del producto.
    df["U_Rel"] = df["Unidades_Vendidas"] / df.groupby("Producto_ID")["Unidades_Vendidas"].transform("mean")
    return df, metas


def fmt_moneda(v: float) -> str:
    return f"${v / 1_000_000:,.2f}M" if abs(v) >= 1_000_000 else f"${v:,.0f}"


def delta_pct(serie: pd.Series, ref: int, prev: int):
    if ref not in serie.index or prev not in serie.index or serie[prev] == 0:
        return None
    return f"{(serie[ref] / serie[prev] - 1) * 100:+.1f}% vs {prev}"


if DATA_FILE is None:
    st.error(f"No encontré el CSV de demanda (Farmacias_Valle_Demanda*.csv) en la carpeta de app.py: {BASE}")
    st.stop()

df_raw, metas = cargar_datos(DATA_FILE, METAS_FILE)

anios_disp = sorted(df_raw["Anio"].unique().tolist())
cats_disp = sorted(df_raw["Categoria"].unique().tolist())
temps_disp = [t for t in ["Alta", "Media", "Baja"] if t in df_raw["Temporada"].unique()]

# --- Filtros (vacío = todas) ---
f1, f2, f3 = st.columns([1.2, 1.4, 1.1])
with f1:
    rango = st.select_slider("Rango de años", options=anios_disp, value=(anios_disp[0], anios_disp[-1]))
with f2:
    cats_sel = st.multiselect("Categorías", cats_disp, default=[], placeholder="Todas las categorías")
with f3:
    temps_sel = st.multiselect("Temporadas", temps_disp, default=[], placeholder="Todas las temporadas")

cats_sel = cats_sel or cats_disp
temps_sel = temps_sel or temps_disp

df_base = df_raw[df_raw["Categoria"].isin(cats_sel) & df_raw["Temporada"].isin(temps_sel)]
df_f = df_base[(df_base["Anio"] >= rango[0]) & (df_base["Anio"] <= rango[1])]

if df_f.empty:
    st.warning("No hay registros con esta combinación de filtros.")
    st.stop()

# --- Cálculos ---
anio_ref, anio_prev = rango[1], rango[1] - 1
ing_y = df_base.groupby("Anio")["Ingreso_Mensual"].sum()
uni_y = df_base.groupby("Anio")["Unidades_Vendidas"].sum()
cob_y = df_base.groupby("Anio")["Cobertura_Meses"].mean()
bajo_y = df_base.groupby("Anio")["Bajo_Meta"].mean() * 100
meta_ref = float(metas.loc[metas["Anio"] == anio_ref, "Cobertura_Meta_Meses"].iloc[0])

cat_df = df_f.groupby("Categoria", as_index=False)["Ingreso_Mensual"].sum().sort_values("Ingreso_Mensual", ascending=False)
cat_df["Pct"] = cat_df["Ingreso_Mensual"] / cat_df["Ingreso_Mensual"].sum() * 100

serie = df_f.groupby("Fecha", as_index=False)["Ingreso_Mensual"].sum().sort_values("Fecha")
idx_mes = serie.groupby(serie["Fecha"].dt.month)["Ingreso_Mensual"].mean() / serie["Ingreso_Mensual"].mean()
dic_pct = (idx_mes[12] - 1) * 100 if (12 in idx_mes.index and len(serie) >= 12) else None

con = df_f.loc[df_f["Promocion"] == 1, "U_Rel"]
sin = df_f.loc[df_f["Promocion"] == 0, "U_Rel"]
lift = (con.mean() / sin.mean() - 1) * 100 if len(con) and len(sin) else None
# Error estándar aproximado del efecto (%): si |efecto| < 2 EE no se distingue de ruido.
ee_lift = (np.sqrt(con.var() / len(con) + sin.var() / len(sin)) / sin.mean() * 100) if len(con) > 1 and len(sin) > 1 else np.inf
pct_bajo = df_f["Bajo_Meta"].mean() * 100

# --- Título narrativo (cambia con los filtros) ---
partes = []
if len(cat_df) >= 2:
    partes.append(f"{cat_df.iloc[0]['Categoria']} y {cat_df.iloc[1]['Categoria']} generan {cat_df.iloc[:2]['Pct'].sum():.0f}% del ingreso")
else:
    partes.append(f"{cat_df.iloc[0]['Categoria']} suma {fmt_moneda(cat_df.iloc[0]['Ingreso_Mensual'])} de ingreso en el periodo")
if lift is not None:
    partes.append(
        f"la promoción no muestra efecto en unidades ({lift:+.1f}%)" if abs(lift) < 2 * ee_lift
        else f"la promoción cambia las unidades en {lift:+.1f}%"
    )
partes.append(f"{pct_bajo:.0f}% de los meses-producto queda bajo la meta de cobertura")
titulo = "; ".join(partes[:-1]) + " y " + partes[-1] if len(partes) > 1 else partes[0]
st.markdown(f'<div class="titulo">{titulo}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Decisión: dónde reforzar el abasto (Abasto) y si conviene seguir promocionando (Comercial) · '
    f"Dirección Comercial y Abasto · KPIs: {anio_ref} vs {anio_prev} · Gráficas: {rango[0]}–{rango[1]}</div>",
    unsafe_allow_html=True,
)

# --- KPIs ---
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Ingreso {anio_ref}", fmt_moneda(ing_y.get(anio_ref, 0)), delta=delta_pct(ing_y, anio_ref, anio_prev),
          help="Suma de ingreso mensual del último año del rango, con los filtros activos.")
k2.metric(f"Unidades vendidas {anio_ref}", f"{uni_y.get(anio_ref, 0):,.0f}", delta=delta_pct(uni_y, anio_ref, anio_prev))
k3.metric(f"Cobertura de stock {anio_ref}", f"{cob_y.get(anio_ref, np.nan):.2f} meses",
          delta=f"{cob_y.get(anio_ref, np.nan) - meta_ref:+.2f} vs meta {meta_ref:.2f}", delta_color="off",
          help="Cobertura = stock disponible ÷ unidades vendidas del mismo mes. Sobre la meta implica inventario de más.")
d_bajo = f"{bajo_y[anio_ref] - bajo_y[anio_prev]:+.1f} pp vs {anio_prev}" if anio_prev in bajo_y.index else None
k4.metric(f"Meses-producto bajo meta {anio_ref}", f"{bajo_y.get(anio_ref, np.nan):.0f}%", delta=d_bajo, delta_color="inverse",
          help="Porcentaje de registros producto-mes con cobertura menor a la meta (riesgo de desabasto).")

st.divider()

# --- Gráficas ---
g1, g2, g3, g4 = st.columns([1.3, 1.05, 1.1, 0.95])
cfg = {"displayModeBar": False}
H = 250


def base_layout(fig, titulo_g, **kw):
    fig.update_layout(
        title=dict(text=f"<b>{titulo_g}</b>", font=dict(size=14), x=0, xanchor="left"),
        height=H, margin=dict(l=5, r=10, t=42, b=5), showlegend=False, **kw,
    )
    return fig


# 1. Tendencia + estacionalidad
with g1:
    fig1 = go.Figure(go.Scatter(x=serie["Fecha"], y=serie["Ingreso_Mensual"], mode="lines",
                                line=dict(color=AZUL, width=2), hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>"))
    decs = serie[serie["Fecha"].dt.month == 12]
    if not decs.empty:
        fig1.add_trace(go.Scatter(x=decs["Fecha"], y=decs["Ingreso_Mensual"], mode="markers",
                                  marker=dict(color=AMBAR, size=8), hoverinfo="skip"))
        ult = decs.iloc[-1]
        fig1.add_annotation(x=ult["Fecha"], y=ult["Ingreso_Mensual"], text="Cada diciembre es el pico",
                            showarrow=True, arrowhead=1, ax=-70, ay=-14, font=dict(size=10, color="#111827"),
                            bgcolor="#fde68a", bordercolor=AMBAR)
    t1 = f"Ingreso mensual: dic. {dic_pct:+.0f}% sobre el promedio" if dic_pct is not None else "Ingreso mensual"
    base_layout(fig1, t1, hovermode="x unified",
                yaxis=dict(range=[0, serie["Ingreso_Mensual"].max() * 1.2], tickprefix="$", tickformat=".2s"),
                xaxis=dict(title=None))
    st.plotly_chart(fig1, width="stretch", config=cfg)

# 2. Comparación por categoría
with g2:
    d2 = cat_df.sort_values("Ingreso_Mensual", ascending=True)
    top2 = set(cat_df["Categoria"].head(2)) if len(cat_df) >= 2 else set(cat_df["Categoria"])
    fig2 = go.Figure(go.Bar(
        x=d2["Ingreso_Mensual"], y=d2["Categoria"], orientation="h",
        marker_color=[AZUL if c in top2 else GRIS for c in d2["Categoria"]],
        text=[f"{fmt_moneda(v)} · {p:.0f}%" for v, p in zip(d2["Ingreso_Mensual"], d2["Pct"])],
        textposition="outside", cliponaxis=False, textfont_size=10, hoverinfo="skip",
    ))
    t2 = f"{' y '.join(cat_df['Categoria'].head(2))} suman {cat_df['Pct'].head(2).sum():.0f}%" if len(cat_df) >= 2 else "Ingreso por categoría"
    base_layout(fig2, t2, xaxis=dict(visible=False, range=[0, d2["Ingreso_Mensual"].max() * 1.6]))
    st.plotly_chart(fig2, width="stretch", config=cfg)

# 3. Distribución de cobertura
with g3:
    fig3 = go.Figure()
    bins = dict(start=1.3, end=2.0, size=0.05)
    for flag, color, nombre in [(True, AMBAR, "Bajo meta"), (False, GRIS, "En o sobre meta")]:
        vals = df_f.loc[df_f["Bajo_Meta"] == flag, "Cobertura_Meses"]
        fig3.add_trace(go.Histogram(x=vals, xbins=bins, marker_color=color, name=nombre,
                                    hovertemplate="%{x:.2f} meses: %{y} registros<extra></extra>"))
    fig3.add_vline(x=meta_ref, line_dash="dash", line_color="#e5e7eb", line_width=1.2,
                   annotation_text=f"Meta {meta_ref:.2f}", annotation_position="top left", annotation_font_size=10)
    base_layout(fig3, f"{pct_bajo:.0f}% de los meses-producto, bajo meta", barmode="overlay", bargap=0.05,
                xaxis=dict(title="Cobertura (meses de stock)", range=[1.28, 2.02]),
                yaxis=dict(title="Registros producto-mes"))
    st.plotly_chart(fig3, width="stretch", config=cfg)

# 4. Efecto de la promoción (KPI destacado)
with g4:
    if lift is None:
        fig4 = go.Figure()
        fig4.add_annotation(text="El filtro no tiene meses<br>con y sin promoción", showarrow=False)
        t4 = "Efecto de la promoción"
        fig4.update_xaxes(visible=False); fig4.update_yaxes(visible=False)
    else:
        idx_con = con.mean() / sin.mean() * 100
        fig4 = go.Figure(go.Bar(
            x=[f"Sin promo<br>(n={len(sin):,})", f"Con promo<br>(n={len(con):,})"], y=[100, idx_con],
            marker_color=[GRIS, AZUL], text=["100", f"{idx_con:.1f}"], textposition="outside", cliponaxis=False,
            hoverinfo="skip"))
        t4 = f"Con promo se vende {lift:+.1f}% vs sin promo"
        fig4.update_yaxes(range=[0, 130], title="Índice (sin promo = 100)")
    base_layout(fig4, t4)
    st.plotly_chart(fig4, width="stretch", config=cfg)

st.markdown(
    f'<div class="pie">Fuente: {DATA_FILE.name} (1,200 registros, 20 productos, 5 categorías, 2020–2024) + tabla de metas (metas_anuales.csv). '
    "Cobertura = stock ÷ unidades del mismo mes. Efecto de promoción: unidades relativas al promedio de cada producto. "
    "Limitaciones: promoción binaria (sin % de descuento), sin tienda/región ni costos, meta de cobertura constante, sin quiebres registrados.</div>",
    unsafe_allow_html=True,
)
