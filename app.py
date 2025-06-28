# app.py ───────────────────────────────────────────────────────────────
"""
DenoiseSH – Dashboard ultra intuitivo y minimalista

Pestañas:
  • 🚀 Inicio              – resumen y estado general
  • 🛠️ Modelos            – subir/reemplazar modelos (.keras + .joblib)
  • 📂 Datos               – métricas y exploración dataset
  • 🧪 Inferencia Manual   – sube imágenes o selecciona batch aleatorio
  • 🔁 Inferencia Continua – loop automático de pruebas
  • ✅ Evaluación          – accuracy con imágenes ground-truth
"""
from __future__ import annotations
import time, random
import streamlit as st
from pathlib import Path
import modelsFunction as mf

_rerun = st.rerun if hasattr(st, "rerun") else st.experimental_rerun

MODELS_DIR, DATASET_DIR = mf.MODELS_DIR, mf.DATASET_DIR
OK_COLOR, ERR_COLOR     = "#21BA45", "#DB2828"

mf.auto_load_models()
st.set_page_config(
    page_title="DenoiseSH",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────── SIDEBAR – ESTADO GLOBAL ────────────────────────────
with st.sidebar:
    st.title("⚓ DenoiseSH")
    st.info(
        "👋 **Bienvenido** a Denoise the Salmon Health. Pasos sugeridos:\n"
        "1️⃣ Sube tus modelos.\n"
        "2️⃣ Explora tu dataset.\n"
        "3️⃣ Realiza pruebas e infiere.\n"
        "4️⃣ Evalúa la precisión."
    )
    n_iters = st.slider("Iteraciones estocásticas", 1, 50, 10, help="Mayor robustez en predicción")
    thr     = st.slider("Umbral de decisión", 0.0, 1.0, 0.50, 0.01, help="Ajusta el umbral del clasificador")
    st.divider()
    enc_ok = mf._encoder is not None
    clf_ok = mf._clf     is not None
    status_col = OK_COLOR if (enc_ok and clf_ok) else ERR_COLOR
    st.markdown(
        f"<b>Estado del sistema:</b> "
        f"<span style='color:{status_col};font-size:1.2em'>"
        f"{'✅ Listo para usar' if (enc_ok and clf_ok) else '❌ Falta cargar modelos'}</span>",
        unsafe_allow_html=True
    )

# ─────────────── PESTAÑAS – NUEVA ORGANIZACIÓN ──────────────────────
tab_home, tab_models, tab_data, tab_manual, tab_loop, tab_eval = st.tabs([
    "🚀 Inicio", "🛠️ Modelos", "📂 Datos", "🧪 Inferencia Manual", "🔁 Inferencia Continua", "✅ Evaluación"
])

# ─────────────── 0. INICIO – GUÍA Y ESTADO ───────────────────────────
with tab_home:
    st.header("🚀 DenoiseSH – Diagnóstico SalmonScan")
    st.markdown("""
    <ul>
      <li><b>¿Nuevo?</b> Empieza en <span style='color:#1e90ff'>🛠️ Modelos</span> y sube tus archivos entrenados.</li>
      <li>Explora el dataset en <span style='color:#1e90ff'>📂 Datos</span>.</li>
      <li>Prueba tu IA en <span style='color:#1e90ff'>🧪 Inferencia Manual</span> o <span style='color:#1e90ff'>🔁 Inferencia Continua</span>.</li>
      <li>Evalúa el rendimiento en <span style='color:#1e90ff'>✅ Evaluación</span>.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("📊 Resumen rápido")
    enc_ok = mf._encoder is not None
    clf_ok = mf._clf     is not None
    h_cnt = len(list((DATASET_DIR/"FreshSalmon").glob("*")))
    i_cnt = len(list((DATASET_DIR/"InfectedSalmon").glob("*")))
    total = h_cnt + i_cnt
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Encoder", "Cargado" if enc_ok else "No cargado", help="Red encoder de autoencoder")
    m2.metric("Clasificador", "Cargado" if clf_ok else "No cargado", help="Clasificador .joblib")
    m3.metric("Imágenes Healthy", h_cnt)
    m4.metric("Imágenes Infected", i_cnt)
    st.caption("¿Listo para comenzar? Ve a la pestaña de Modelos o explora tus datos.")

# ─────────────── 1. MODELOS – CARGA Y ESTADO ────────────────────────
with tab_models:
    st.header("🛠️ Subir o reemplazar modelos entrenados")
    st.markdown("""
    <b>Paso 1:</b> Sube tu <u>encoder</u> (.keras) y <u>clasificador</u> (.joblib).<br>
    <b>Paso 2:</b> (Opcional) Sube un decoder (.keras) para visualizar reconstrucciones.<br>
    <b>Ambos modelos obligatorios para usar el sistema.</b>
    """, unsafe_allow_html=True)
    st.divider()
    c1, c2 = st.columns(2)
    if enc_ok: c1.success("✅ Encoder cargado")
    else:      c1.warning("⚠️ Encoder NO cargado")
    if clf_ok: c2.success("✅ Clasificador cargado")
    else:      c2.warning("⚠️ Clasificador NO cargado")
    st.divider()
    # --- ZIP uploader ---
    zip_file = st.file_uploader("Subir ZIP con encoder.keras y classifier.joblib", type="zip", key="zip_upl")
    if zip_file and st.button("Cargar modelos desde ZIP"):
        try:
            mf.load_models_from_zip(zip_file)
            st.success("🎉 Modelos cargados correctamente desde ZIP.")
            _rerun()
        except Exception as e:
            st.error(f"Error al cargar ZIP: {e}")

    st.markdown("<b style='color:#bbb'>O sube archivos sueltos:</b>", unsafe_allow_html=True)
    enc_file = st.file_uploader("Encoder (.keras) [Obligatorio]", ["keras"], key="enc_upl")
    dec_file = st.file_uploader("Decoder (.keras) (Opcional)", ["keras"], key="dec_upl")
    clf_file = st.file_uploader("Clasificador (.joblib) [Obligatorio]", ["joblib"], key="clf_upl")

    if st.button("💾 Cargar modelos sueltos"):
        if enc_file and clf_file:
            MODELS_DIR.mkdir(exist_ok=True)
            (MODELS_DIR/"encoder.keras").write_bytes(enc_file.read())
            (MODELS_DIR/"classifier.joblib").write_bytes(clf_file.read())
            mf.load_encoder(MODELS_DIR/"encoder.keras")
            mf.load_classifier(MODELS_DIR/"classifier.joblib")
            if dec_file:
                (MODELS_DIR/"decoder.keras").write_bytes(dec_file.read())
                mf.load_decoder(MODELS_DIR/"decoder.keras")
            st.success("🎉 Modelos cargados correctamente.")
            _rerun()
        else:
            st.error("Debes subir *encoder* y *clasificador* para continuar.")

    st.info(
        "¿Necesitas modelos de ejemplo o ayuda sobre el formato? "
        "Contacta a tu admin o descarga ejemplos en [este enlace](#)."
    )

# ─────────────── helper: caption HTML ───────────────────────────────
def caption_html(true_lbl: int, pred: int, prob: float) -> str:
    true_txt = "Infected" if true_lbl else "Healthy"
    pred_txt = "Infected" if pred else "Healthy"
    ok       = (true_lbl == pred)
    tag_txt, tag_col = ("CORRECTO", OK_COLOR) if ok else ("ERROR", ERR_COLOR)
    return (
        f"<div style='text-align:center;font-size:.85rem'>"
        f"<b>Real:</b> {true_txt}<br>"
        f"<b>Pred:</b> {pred_txt}<br>"
        f"<b>P={prob:.2f}</b><br>"
        f"<span style='color:{tag_col};font-weight:600'>{tag_txt}</span>"
        f"</div>"
    )

# ─────────────── 2. 📂 DATOS – DASHBOARD & EXPLORACIÓN ──────────────
with tab_data:
    st.header("📂 Datos – Exploración del dataset")
    if not (enc_ok and clf_ok):
        st.warning("Carga primero tus modelos para explorar los datos.")
        st.stop()
    h_cnt = len(list((DATASET_DIR/"FreshSalmon").glob("*")))
    i_cnt = len(list((DATASET_DIR/"InfectedSalmon").glob("*")))
    total = h_cnt + i_cnt
    ratio = i_cnt / total if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Healthy",  h_cnt)
    m2.metric("Infected", i_cnt)
    m3.metric("Total",    total)
    m4.metric("Proporción infectados", f"{ratio:.1%}" if total else "–")

    st.divider()
    st.subheader("👀 Muestra aleatoria balanceada del dataset")
    cc = st.columns([1,1,1])
    num_h = cc[0].number_input("Healthy",  1, 100, 4, key="m_h2")
    num_i = cc[1].number_input("Infected", 1, 100, 2, key="m_i2")
    if cc[2].button("Ver muestra"):
        paths = mf.sample_images(num_h, num_i)
        cols  = st.columns(min(len(paths), 8))
        for idx, p in enumerate(paths):
            tl, prob, pred, _ = (
                mf.infer_label_from_path(p),
                *mf.stochastic_predict(mf.preprocess_image(p), n_iters, thr)
            )
            with cols[idx % len(cols)]:
                st.image(str(p), width=120)
                st.markdown(caption_html(tl, pred, prob), unsafe_allow_html=True)

# ─────────────── 3. 🧪 INFERENCIA MANUAL – SUBE/SELECCIONA ───────────
with tab_manual:
    st.header("🧪 Inferencia Manual")
    if not (enc_ok and clf_ok):
        st.warning("Carga primero tus modelos para ejecutar inferencias.")
        st.stop()
    st.subheader("1️⃣ Subir imágenes (multi-upload)")
    uploads = st.file_uploader(
        "Selecciona imágenes", ["jpg","jpeg","png"], accept_multiple_files=True
    )
    if uploads:
        cols = st.columns(min(len(uploads), 5))
        for idx, up in enumerate(uploads):
            prob, pred, _ = mf.stochastic_predict(mf.pil_to_tensor(up), n_iters, thr)
            with cols[idx % len(cols)]:
                st.image(up, width=120)
                st.markdown(f"<b>Pred:</b> {'Infected' if pred else 'Healthy'}<br><b>P={prob:.2f}</b>", unsafe_allow_html=True)

    st.divider()
    st.subheader("2️⃣ Batch aleatorio del dataset")
    ccb = st.columns([1,1,1])
    b_h = ccb[0].number_input("Healthy",  1, 200, 4, key="bat_h2")
    b_i = ccb[1].number_input("Infected", 1, 200, 2, key="bat_i2")
    if ccb[2].button("Generar batch aleatorio"):
        try:
            paths = mf.sample_images(b_h, b_i)
            cols  = st.columns(min(len(paths), 8))
            for idx, p in enumerate(paths):
                tl, prob, pred, _ = (
                    mf.infer_label_from_path(p),
                    *mf.stochastic_predict(mf.preprocess_image(p), n_iters, thr)
                )
                with cols[idx % len(cols)]:
                    st.image(str(p), width=120)
                    st.markdown(caption_html(tl, pred, prob), unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

# ─────────────── 4. 🔁 INFERENCIA CONTINUA – LOOP AUTOMÁTICO ────────
with tab_loop:
    st.header("🔁 Inferencia Continua (loop automático)")
    if not (enc_ok and clf_ok):
        st.warning("Carga primero tus modelos para ejecutar el loop.")
        st.stop()
    for key in ("loop_on","loop_total","loop_hits"):
        st.session_state.setdefault(key, False if key=="loop_on" else 0)
    c_start, c_stop, _ = st.columns(3)
    if c_start.button("▶️ Iniciar loop") and not st.session_state.loop_on:
        st.session_state.loop_on    = True
        st.session_state.loop_total = 0
        st.session_state.loop_hits  = 0
        _rerun()
    if c_stop.button("⏹️ Detener loop") and st.session_state.loop_on:
        st.session_state.loop_on = False
        _rerun()
    if st.session_state.loop_on:
        img_ph  = st.empty()
        info_ph = st.empty()
        m1, m2  = st.columns(2)
        pool = (
            list((DATASET_DIR/"FreshSalmon").glob("*")) +
            list((DATASET_DIR/"InfectedSalmon").glob("*"))
        )
        if not pool:
            st.warning("dataset/ vacío")
            st.session_state.loop_on = False
        else:
            p = random.choice(pool)
            tl, prob, pred, _ = (
                mf.infer_label_from_path(p),
                *mf.stochastic_predict(mf.preprocess_image(p), n_iters, thr)
            )
            hit = (tl == pred)
            st.session_state.loop_total += 1
            st.session_state.loop_hits  += int(hit)
            img_ph.image(str(p), width=170)
            info_ph.markdown(caption_html(tl, pred, prob), unsafe_allow_html=True)
            acc = st.session_state.loop_hits / st.session_state.loop_total
            m1.metric("Procesadas", st.session_state.loop_total)
            m2.metric("Accuracy loop", f"{acc:.1%}")
            time.sleep(2)
            _rerun()

# ─────────────── 5. ✅ EVALUACIÓN – ACCURACY Y VERIFICACIÓN ──────────
with tab_eval:
    st.header("✅ Evaluación (accuracy y verificación)")
    if not (enc_ok and clf_ok):
        st.warning("Carga primero tus modelos para evaluar resultados.")
        st.stop()
    st.session_state.setdefault("ver_hits", 0)
    st.session_state.setdefault("ver_tot",  0)
    st.subheader("Verificación rápida")
    if st.button("🔄 Reiniciar métricas"):
        st.session_state.ver_hits = st.session_state.ver_tot = 0

    def _log(hit: bool):
        st.session_state.ver_tot  += 1
        st.session_state.ver_hits += int(hit)

    mode = st.radio("Tipo de verificación", ["Subir imagen", "Aleatoria del dataset"], horizontal=True)
    if mode == "Subir imagen":
        upv = st.file_uploader("Imagen a verificar", ["jpg","jpeg","png"], key="v_up")
        true_lbl = 0 if st.radio("Etiqueta real", ["Healthy (0)", "Infected (1)"]).startswith("Healthy") else 1
        if st.button("✅ Comprobar imagen"):
            if upv:
                prob, pred, hit, _ = mf.verify_prediction(mf.pil_to_tensor(upv), true_lbl, n_iters, thr)
                _log(hit)
                st.image(upv, caption=caption_html(true_lbl, pred, prob), use_container_width=True)
    else:
        if st.button("🎲 Verificar imagen aleatoria"):
            pool = (
                list((DATASET_DIR/"FreshSalmon").glob("*")) +
                list((DATASET_DIR/"InfectedSalmon").glob("*"))
            )
            if pool:
                p = random.choice(pool)
                true_lbl = mf.infer_label_from_path(p)
                prob, pred, hit, _ = mf.verify_prediction(mf.preprocess_image(p), true_lbl, n_iters, thr)
                _log(hit)
                st.image(str(p), caption=caption_html(true_lbl, pred, prob), use_container_width=True)
            else:
                st.warning("dataset/ vacío")

    st.divider()
    st.subheader("Verificación por lote")
    vh, vi, btn2 = st.columns([1,1,1])
    v_h = vh.number_input("# Healthy",  1, 200, 5)
    v_i = vi.number_input("# Infected", 1, 200, 5)
    if btn2.button("🧪 Ejecutar lote"):
        try:
            paths = mf.sample_images(v_h, v_i)
            cols  = st.columns(min(len(paths), 8))
            for idx, p in enumerate(paths):
                true_lbl = mf.infer_label_from_path(p)
                prob, pred, hit, _ = mf.verify_prediction(mf.preprocess_image(p), true_lbl, n_iters, thr)
                _log(hit)
                with cols[idx % len(cols)]:
                    st.image(str(p), width=120)
                    st.markdown(caption_html(true_lbl, pred, prob), unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

    if st.session_state.ver_tot:
        acc = st.session_state.ver_hits / st.session_state.ver_tot
        st.markdown(
            f"<h4>Accuracy acumulado: "
            f"<span style='color:{OK_COLOR if acc>=0.5 else ERR_COLOR}'>"
            f"{acc:.1%}</span> "
            f"({st.session_state.ver_hits}/{st.session_state.ver_tot})</h4>",
            unsafe_allow_html=True
        )
