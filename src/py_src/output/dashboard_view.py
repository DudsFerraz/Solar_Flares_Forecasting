import streamlit as st
import json
import time
import os
from dotenv import load_dotenv

st.set_page_config(
    page_title="Solar Flare Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("☀️ Monitor de Previsão de Explosões Solares")
st.markdown("---")

load_dotenv()
JSON_PATH = os.getenv('DASHBOARD_JSON_PATH')

def load_data():
    if not os.path.exists(JSON_PATH):
        return None
    try:
        with open(JSON_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


data = load_data()

if data is None:
    st.warning("⏳ Aguardando dados do pipeline... (Arquivo JSON ainda não foi gerado)")
    st.info("Certifique-se de que o 'main_pipeline.py' está rodando em outro terminal.")

    if st.button("Tentar Novamente"):
        st.rerun()
    st.stop()

last_update = data.get('last_update_utc', 'N/A').replace('T', ' ')[:19]
latency = data.get('data_latency_minutes', 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Última Atualização (UTC)", last_update)
with col2:
    st.metric(
        "Latência dos Dados",
        f"{latency:.1f} min",
        delta="Normal" if latency < 15 else "Atraso Detectado",
        delta_color="normal" if latency < 15 else "inverse"
    )
with col3:
    st.metric("Status do Sistema", "OPERACIONAL 🟢")

st.markdown("### 🔮 Previsões por Janela de Tempo")

predictions = data.get('predictions', {})

if not predictions:
    st.error("O arquivo de dados existe, mas está vazio ou sem previsões.")
else:
    cols = st.columns(len(predictions))

    for idx, (window, res) in enumerate(predictions.items()):
        with cols[idx]:
            cls_final = res.get('final_class', 'N/A')
            prob = res.get('probability', 0.0)
            risk = res.get('risk_level', 'Unknown')
            flux = res.get('estimated_flux')
            msg = res.get('msg')

            st.subheader(f"⏱️ {window}")

            if msg:
                st.error(f"Erro: {msg}")
            else:
                if "No Flare" in cls_final:
                    st.success(f"**{cls_final}**")
                elif "Class A" in cls_final or "Class B" in cls_final:
                    st.success(f"**{cls_final}**")
                elif "Class C" in cls_final:
                    st.warning(f"**{cls_final}**")
                elif "Class M" in cls_final:
                    st.warning(f"**{cls_final}** (Alto Risco)")
                elif "Class X" in cls_final:
                    st.error(f"🚨 **{cls_final}** (CRÍTICO)")
                else:
                    st.info(f"**{cls_final}**")

                st.progress(min(max(float(prob), 0.0), 1.0))
                st.caption(f"Confiança do Modelo: {prob:.1%}")

                st.markdown(f"**Risco:** {risk}")

                if flux:
                    st.markdown(f"**Fluxo Est.:** `{flux:.2e} W/m²`")

st.markdown("---")
col_refresh, col_info = st.columns([1, 4])
with col_refresh:
    if st.button('🔄 Atualizar Agora'):
        st.rerun()
with col_info:
    st.caption("O dashboard lê o arquivo local gerado pelo pipeline de backend.")

time.sleep(60)
st.rerun()