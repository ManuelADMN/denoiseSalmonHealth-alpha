# modelsFunction.py ────────────────────────────────────────────────────
# Funciones compartidas con app.py (Streamlit):
#   · Carga de modelos   · Pre-procesado  · Predicción estocástica
#   · Sampling balanceado · Verificación con ground-truth
# ---------------------------------------------------------------------

from __future__ import annotations
import pathlib, random, warnings
from typing import Tuple, List, Optional
import zipfile
import io
import numpy as np
from PIL import Image
import tensorflow as tf                 # TF ≥ 2.9
import joblib

# ──────────── Rutas y constantes globales ─────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"          # encoder.keras · classifier.joblib
DATASET_DIR = BASE_DIR / "dataset"         # …/FreshSalmon / InfectedSalmon

IMG_H, IMG_W = 250, 600                   # tamaño usado al entrenar

# ──────────── Estado en memoria (modelos cargados) ───────────────────
_encoder: Optional[tf.keras.Model] = None
_decoder: Optional[tf.keras.Model] = None
_clf: Optional[object]             = None      # RandomForest, MLP, etc.

# ──────────── Custom-objects para load_model (ej. GaussianNoise) ─────
_CUSTOM_OBJECTS = {
    "GaussianNoise": tf.keras.layers.GaussianNoise,
}

# ──────────── Importación de modelos (zip) ────────────────────
def load_models_from_zip(zipfile_obj):
    """
    Carga encoder.keras y classifier.joblib/.pkl desde un ZIP subido.
    zipfile_obj puede ser un archivo BytesIO (ej: de st.file_uploader)
    """
    global _encoder, _clf, _decoder
    # Asegura carpeta destino
    MODELS_DIR.mkdir(exist_ok=True)
    # Abre el ZIP en memoria o desde archivo
    with zipfile.ZipFile(zipfile_obj) as zf:
        enc_names = [n for n in zf.namelist() if n.endswith("encoder.keras")]
        clf_names = [n for n in zf.namelist() if n.endswith("classifier.joblib") or n.endswith("classifier.pkl")]
        dec_names = [n for n in zf.namelist() if n.endswith("decoder.keras")]
        if not enc_names or not clf_names:
            raise FileNotFoundError("El ZIP debe contener encoder.keras y classifier.joblib (o .pkl)")
        # Extrae archivos a la carpeta MODELS_DIR
        enc_out = MODELS_DIR / "encoder.keras"
        for n in enc_names:
            with zf.open(n) as f, open(enc_out, "wb") as out:
                out.write(f.read())
        # Busca el primer clasificador encontrado
        clf_src = clf_names[0]
        clf_out = MODELS_DIR / ("classifier.joblib" if clf_src.endswith("joblib") else "classifier.pkl")
        with zf.open(clf_src) as f, open(clf_out, "wb") as out:
            out.write(f.read())
        # Opcional: decoder
        if dec_names:
            dec_out = MODELS_DIR / "decoder.keras"
            with zf.open(dec_names[0]) as f, open(dec_out, "wb") as out:
                out.write(f.read())
        # Carga modelos normalmente
        load_encoder(enc_out)
        load_classifier(clf_out)
        if dec_names:
            load_decoder(MODELS_DIR / "decoder.keras")
    print("[modelsFunction] Modelos cargados desde ZIP.")

# ──────────── Carga de modelos ───────────────────────────────────────
def load_encoder(path: str | pathlib.Path = MODELS_DIR / "encoder.keras"):
    global _encoder
    _encoder = tf.keras.models.load_model(path, compile=False,
                                          custom_objects=_CUSTOM_OBJECTS)
    return _encoder

def load_decoder(path: str | pathlib.Path = MODELS_DIR / "decoder.keras"):
    global _decoder
    _decoder = tf.keras.models.load_model(path, compile=False,
                                          custom_objects=_CUSTOM_OBJECTS)
    return _decoder

def load_classifier(path: str | pathlib.Path = MODELS_DIR / "classifier.joblib"):
    global _clf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _clf = joblib.load(path)
    return _clf

# ──────────── Pre-procesado de imágenes ──────────────────────────────
def preprocess_image(path: str | pathlib.Path) -> tf.Tensor:
    """Lee una imagen → tensor float32 normalizado (IMG_H, IMG_W, 3)."""
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    return tf.cast(img, tf.float32) / 255.0

def pil_to_tensor(uploaded_file) -> tf.Tensor:
    """Convierte imagen subida en Streamlit a tensor listo para inferencia."""
    img = Image.open(uploaded_file).convert("RGB").resize((IMG_W, IMG_H))
    return tf.convert_to_tensor(np.asarray(img) / 255.0, dtype=tf.float32)

# ──────────── Predicción estocástica (Dropout activo) ────────────────
def stochastic_predict(img_tensor: tf.Tensor,
                       n_iters: int = 10,
                       thr: float  = 0.5
                      ) -> Tuple[float, int, List[float]]:
    """
    Ejecuta n_iters pasadas con _encoder(training=True).
    Devuelve: media P(Infected), pred_final (0/1), lista completa de probs.
    """
    if _encoder is None or _clf is None:
        raise RuntimeError("Primero carga encoder y clasificador.")
    img_b = tf.expand_dims(img_tensor, 0)
    probs = [
        float(_clf.predict_proba(_encoder(img_b, training=True))[0, 1])
        for _ in range(n_iters)
    ]
    mean = float(np.mean(probs))
    return mean, int(mean >= thr), probs

# ──────────── Verificación contra etiqueta real ──────────────────────
def verify_prediction(img_tensor: tf.Tensor,
                      true_label: int,
                      n_iters: int = 10,
                      thr: float  = 0.5
                     ) -> Tuple[float, int, bool, List[float]]:
    """
    Devuelve: media, pred, hit (=pred==true_label), lista_probs.
    """
    mean, pred, probs = stochastic_predict(img_tensor, n_iters, thr)
    hit = pred == true_label
    return mean, pred, hit, probs

def infer_label_from_path(path: pathlib.Path) -> Optional[int]:
    """Deducción simple según nombre de carpeta."""
    p = path.parent.name.lower()
    if "infect" in p:
        return 1
    if "fresh" in p:
        return 0
    return None

# ──────────── Sampling balanceado (Healthy / Infected) ───────────────
def _paths_by_class() -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
    healthy = list((DATASET_DIR / "FreshSalmon").glob("*"))
    infect  = list((DATASET_DIR / "InfectedSalmon").glob("*"))
    return healthy, infect

def sample_images(n_healthy: int, n_infected: int) -> List[pathlib.Path]:
    h, i = _paths_by_class()
    if n_healthy > len(h) or n_infected > len(i):
        raise ValueError("No hay suficientes imágenes en alguna clase.")
    paths = random.sample(h, n_healthy) + random.sample(i, n_infected)
    random.shuffle(paths)
    return paths

# ──────────── Carga automática al importar módulo ────────────────────
def auto_load_models():
    enc_p = MODELS_DIR / "encoder.keras"
    clf_p = None
    for ext in ("classifier.joblib", "classifier.pkl"):
        path = MODELS_DIR / ext
        if path.exists():
            clf_p = path
            break
    if enc_p.exists() and clf_p:
        try:
            load_encoder(enc_p)
            load_classifier(clf_p)
            print("[modelsFunction] Modelos autocargados.")
        except Exception as e:
            print("[modelsFunction] ⚠️  Falló autocarga:", e)

auto_load_models()
