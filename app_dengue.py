# Vigilancia epidemiologica del dengue - Ica (DATOS SIMULADOS con fines educativos)
# Ejecutar: python -m streamlit run app_dengue.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Vigilancia de Dengue - Ica", page_icon="🦟", layout="wide")

# Poblacion referencial SIMULADA por distrito (no son cifras oficiales)
DISTRITOS = {
    "Ica": 130000, "La Tinguiña": 38000, "Parcona": 58000, "Subtanjalla": 36000,
    "Los Aquijes": 21000, "Santiago": 31000, "Pueblo Nuevo": 6000, "Salas": 19000,
}
GRUPOS_EDAD = ["0-11", "12-17", "18-29", "30-59", "60+"]


# ---------------------------------------------------------------- DATOS
@st.cache_data
def generar_datos(semilla=42):
    """Genera casos semanales simulados con estacionalidad (pico en verano)."""
    rng = np.random.default_rng(semilla)
    semanas = pd.date_range("2024-01-01", "2026-10-26", freq="W-MON")
    filas = []
    for d, pob in DISTRITOS.items():
        factor = rng.uniform(0.6, 1.4)
        for f in semanas:
            estacion = 1 + 0.9 * np.sin((f.dayofyear - 60) / 365 * 2 * np.pi)
            brote = 2.5 if (f.year == 2025 and 5 <= f.month <= 6) else 1.0
            esperado = pob / 100000 * 6 * factor * max(estacion, 0.05) * brote
            filas.append((f, d, pob, rng.poisson(esperado)))
    df = pd.DataFrame(filas, columns=["Fecha", "Distrito", "Poblacion", "Casos"])
    df["Año"] = df["Fecha"].dt.year
    df["Semana"] = df["Fecha"].dt.isocalendar().week.astype(int)
    return df


@st.cache_data
def generar_perfil(df_casos, semilla=7):
    """Reparte los casos por edad y sexo (simulado)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for (anio, dist), g in df_casos.groupby(["Año", "Distrito"]):
        total = int(g["Casos"].sum())
        if total == 0:
            continue
        cuentas = rng.multinomial(total, [0.12, 0.10, 0.28, 0.38, 0.12])
        for edad, n in zip(GRUPOS_EDAD, cuentas):
            h = rng.binomial(n, 0.48)
            filas.append((anio, dist, edad, "Hombre", h))
            filas.append((anio, dist, edad, "Mujer", n - h))
    return pd.DataFrame(filas, columns=["Año", "Distrito", "Edad", "Sexo", "Casos"])


# ---------------------------------------------------------------- ENTRADA
st.sidebar.title("🦟 Filtros")
archivo = st.sidebar.file_uploader(
    "Cargar tus datos (CSV: Fecha, Distrito, Poblacion, Casos)", type="csv")
if archivo is not None:
    base = pd.read_csv(archivo, parse_dates=["Fecha"])
    base["Año"] = base["Fecha"].dt.year
    base["Semana"] = base["Fecha"].dt.isocalendar().week.astype(int)
    st.sidebar.success("Usando datos cargados")
else:
    base = generar_datos()

anios = sorted(base["Año"].unique())
anio = st.sidebar.selectbox("Año", anios, index=len(anios) - 1)
distritos = st.sidebar.multiselect(
    "Distritos", sorted(base["Distrito"].unique()), default=sorted(base["Distrito"].unique()))
umbral = st.sidebar.slider("Umbral de alerta (casos por semana)", 5, 100, 25)

df = base[(base["Año"] == anio) & (base["Distrito"].isin(distritos))]
if df.empty:
    st.warning("Selecciona al menos un distrito con datos.")
    st.stop()

# ---------------------------------------------------------------- CALCULOS
poblacion = df.drop_duplicates("Distrito")["Poblacion"].sum()
total_casos = int(df["Casos"].sum())
incidencia = total_casos / poblacion * 100000

semanal = df.groupby("Fecha", as_index=False)["Casos"].sum()
semanal["Media movil (4 sem)"] = semanal["Casos"].rolling(4, min_periods=1).mean()
ultima = int(semanal["Casos"].iloc[-1])
previa = int(semanal["Casos"].iloc[-2]) if len(semanal) > 1 else ultima
delta = ultima - previa

if ultima >= umbral:
    nivel, color = "🔴 ALERTA", "red"
elif ultima >= 0.6 * umbral:
    nivel, color = "🟡 PRECAUCIÓN", "orange"
else:
    nivel, color = "🟢 NORMAL", "green"

inc_dist = (df.groupby(["Distrito", "Poblacion"], as_index=False)["Casos"].sum())
inc_dist["Incidencia x 100 mil"] = (inc_dist["Casos"] / inc_dist["Poblacion"] * 100000).round(1)
inc_dist = inc_dist.sort_values("Incidencia x 100 mil", ascending=False)

# ---------------------------------------------------------------- INTERFAZ
st.title("Vigilancia epidemiológica del dengue - Ica")
st.caption(f"Año {anio} · Datos simulados con fines educativos, no son estadísticas oficiales.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Casos acumulados", f"{total_casos:,}")
c2.metric("Incidencia x 100 mil hab.", f"{incidencia:.1f}")
c3.metric("Casos última semana", ultima, f"{delta:+d} vs semana previa", delta_color="inverse")
c4.metric("Nivel de alerta", nivel)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Tendencia", "🗺️ Distritos", "👥 Perfil", "📋 Datos"])

with tab1:
    fig = go.Figure()
    fig.add_bar(x=semanal["Fecha"], y=semanal["Casos"], name="Casos semanales",
                marker_color="#9ecae1")
    fig.add_scatter(x=semanal["Fecha"], y=semanal["Media movil (4 sem)"],
                    name="Media móvil 4 sem", line=dict(color="#08519c", width=3))
    fig.add_hline(y=umbral, line_dash="dash", line_color="red",
                  annotation_text="Umbral de alerta")
    fig.update_layout(title="Casos de dengue por semana", xaxis_title="Semana",
                      yaxis_title="Casos", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    comp = base[base["Distrito"].isin(distritos)].groupby(["Año", "Semana"], as_index=False)["Casos"].sum()
    fig2 = px.line(comp, x="Semana", y="Casos", color=comp["Año"].astype(str),
                   title="Comparación entre años (por semana epidemiológica)",
                   labels={"color": "Año"})
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    a, b = st.columns(2)
    fig3 = px.bar(inc_dist, x="Incidencia x 100 mil", y="Distrito", orientation="h",
                  color="Incidencia x 100 mil", color_continuous_scale="Reds",
                  title="Incidencia acumulada por distrito")
    fig3.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    a.plotly_chart(fig3, use_container_width=True)

    mapa = df.pivot_table(index="Distrito", columns="Semana", values="Casos", aggfunc="sum")
    fig4 = px.imshow(mapa, aspect="auto", color_continuous_scale="YlOrRd",
                     title="Mapa de calor: casos por distrito y semana",
                     labels=dict(x="Semana", y="Distrito", color="Casos"))
    b.plotly_chart(fig4, use_container_width=True)

with tab3:
    perfil = generar_perfil(base)
    p = perfil[(perfil["Año"] == anio) & (perfil["Distrito"].isin(distritos))]
    a, b = st.columns(2)
    por_edad = p.groupby(["Edad", "Sexo"], as_index=False)["Casos"].sum()
    fig5 = px.bar(por_edad, x="Edad", y="Casos", color="Sexo", barmode="group",
                  category_orders={"Edad": GRUPOS_EDAD}, title="Casos por grupo de edad y sexo")
    a.plotly_chart(fig5, use_container_width=True)
    por_sexo = p.groupby("Sexo", as_index=False)["Casos"].sum()
    fig6 = px.pie(por_sexo, names="Sexo", values="Casos", hole=0.45,
                  title="Distribución por sexo")
    b.plotly_chart(fig6, use_container_width=True)

with tab4:
    st.dataframe(df[["Fecha", "Distrito", "Poblacion", "Casos"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Descargar datos filtrados (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"dengue_ica_{anio}.csv", mime="text/csv")

st.divider()
st.info(
    f"**Interpretación:** en {anio} se registran {total_casos:,} casos "
    f"(incidencia {incidencia:.1f} por 100 mil hab.). La última semana tuvo {ultima} casos, "
    f"nivel {nivel.split(' ', 1)[1].lower()} según el umbral de {umbral} casos/semana."
)