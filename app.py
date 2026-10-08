"""Eikonorasís CIFAR-10 akrivís — clasificación, evaluación y selector desplegable de imágenes."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from src.config import AppConfig, CLASS_NAMES_ES, PROJECT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parent
DIAGRAM_PATH = PROJECT_ROOT / "assets" / "flujo_clasificacion_xception.png"
ASSET_PATH = PROJECT_ROOT / "assets" / "flujo_evaluador_xception.png"
from src.inference import Cifar10Classifier
from src.logging_config import configure_logging
from src.preprocessing import ImageValidationError, validate_uploaded_image

# Recurso gráfico del modelo: ruta absoluta basada en la raíz de la app.
DIAGRAM_PATH = Path(__file__).resolve().parent / "assets" / "modelo_xception_eikonorasis.svg"

TRIVIA_TEXT = '### ¿De dónde viene Eikonorasís?\n\n**Eikonorasís** es un neologismo creado para esta aplicación e inspirado en dos raíces griegas:\n\n- **Eikón (εἰκών):** imagen, figura o representación.\n- **Órasis (ὅρασις):** visión o acto de ver.\n\nEl nombre sintetiza la función de la aplicación: recibir una imagen, procesarla mediante un modelo de visión por computadora y producir una clasificación visual.\n\n### ¿Qué significa *akrivís*?\n\n**Akrivís (ἀκριβής)** alude a precisión, exactitud o rigurosidad. En el proyecto, **Eikonorasís CIFAR-10 akrivís** identifica la versión final del sistema y su evaluador de clasificación.\n\n### ¿Qué representa el logo?\n\n- **El ojo:** percepción y visión.\n- **Los píxeles:** la imagen digital de entrada.\n- **Los nodos conectados:** red neuronal y aprendizaje profundo.\n- **Dorado y grafito:** identidad visual de Eikonorasís.\n\n**CIFAR-10** identifica el conjunto de datos utilizado por el proyecto, con diez categorías oficiales.\n\n**Idea central:** Eikonorasís transforma una imagen digital en una decisión de clasificación mediante visión por computadora.\n'
HELP_TEXT = '### Tecnología\n\nEl evaluador final utiliza **Xception + Transfer Learning** y ajuste fino (*fine-tuning*) para clasificar las diez categorías de CIFAR-10.\n\n**Flujo de inferencia**\n\nImagen → RGB → 160×160 → Xception → Transfer Learning → fine-tuning → 10 clases → predicción y confianza.\n\n### Glosario\n\n- **Xception:** arquitectura convolucional utilizada como modelo base preentrenado.\n- **Transfer Learning:** reutilización de representaciones aprendidas previamente para un nuevo problema.\n- **Fine-tuning:** ajuste controlado de parte del modelo con una tasa de aprendizaje menor.\n- **Inferencia:** ejecución del modelo entrenado sobre una imagen nueva.\n- **Top-3:** tres clases con mayor probabilidad predicha.\n- **Confianza:** probabilidad asociada a la clase predicha.\n'

TRAINING_TEXT = '### Resultados finales del modelo evaluador\n\nEl evaluador final utiliza **Xception + Transfer Learning**, con ajuste fino controlado, para la clasificación de las 10 clases oficiales de CIFAR-10.\n\n**Evaluación formal del modelo:**\n- Validación: **91.10 %**\n- Test CIFAR-10: **90.36 %**\n- Modelo promovido: `model_final.keras`\n\n**Prueba funcional de la aplicación:**\n- Imágenes precargadas procesadas: **30/30**\n- Aciertos: **30/30**\n- Accuracy funcional demostrativa: **100.00 %**\n\nLa prueba de 30 imágenes es funcional y demostrativa; no sustituye la evaluación formal sobre el conjunto de prueba de CIFAR-10.\n\nNo se realizó entrenamiento adicional durante la integración de la aplicación.\n'

MODEL_TEXT = '### Modelo evaluador\n\n**Xception + Transfer Learning** sobre CIFAR-10.\n\n- **Modelo base:** Xception preentrenada con pesos de ImageNet.\n- **Entrada de inferencia:** imagen RGB redimensionada a **160×160×3**.\n- **Transfer Learning:** extracción de representaciones visuales a partir del modelo preentrenado.\n- **Fine-tuning:** ajuste controlado para adaptar el evaluador al problema de 10 clases.\n- **Salida:** 10 categorías oficiales de CIFAR-10.\n- **Inferencia:** clase predicha, confianza y Top-3.\n\nEl modelo utilizado por la aplicación es `model_final.keras`.\n'

DIAGRAM = PROJECT_ROOT / "assets" / "modelo_xception_transfer_learning.svg"
configure_logging()
LOGGER = logging.getLogger(__name__)
CONFIG = AppConfig()
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODEL_DIR = PROJECT_ROOT / 'model'
TEST_IMAGES_DIR = PROJECT_ROOT / 'assets' / 'test_images'
TEST_MANIFEST = TEST_IMAGES_DIR / 'manifest.csv'
st.set_page_config(page_title='Eikonorasís CIFAR-10 akrivís', page_icon='👁️', layout='centered')
st.markdown('\n    <style>\n    :root {\n        --bg:#F7F4EE; --surface:#FFFFFF; --graphite:#23262B;\n        --gold:#C88A12; --gold-soft:#E7BF67; --gold-pale:#F8EED6;\n        --border:#DED8CC; --muted:#62666D;\n    }\n\n    html, body { overflow-x:hidden; }\n\n    .stApp {\n        background:var(--bg);\n        color:var(--graphite);\n    }\n\n    .block-container {\n        width:min(100%,1020px);\n        max-width:1020px;\n        margin:0 auto;\n        padding:5.2rem 1.2rem 2rem 1.2rem;\n    }\n\n    h1,h2,h3 {\n        color:var(--graphite);\n        overflow-wrap:anywhere;\n    }\n\n    h1 {\n        font-size:clamp(2rem,4vw,3rem);\n        line-height:1.08;\n    }\n\n    .kicker {\n        color:#7B570B;\n        font-size:.78rem;\n        font-weight:800;\n        letter-spacing:.08em;\n        text-transform:uppercase;\n    }\n\n    .status,.analysis-box,.result,.preview-box {\n        background:var(--surface);\n        border:1px solid var(--border);\n        border-radius:14px;\n        padding:.9rem 1rem;\n        margin:.8rem 0;\n    }\n\n    .status {\n        background:var(--gold-pale);\n        border-left:6px solid var(--gold);\n    }\n\n    .status-warning {\n        background:#FFF5E5;\n        border-left-color:#A66B00;\n    }\n\n    .status-dark {\n        background:#EFF0F1;\n        border-left-color:var(--graphite);\n    }\n\n    .analysis-box {\n        background:#FFFDF8;\n        border-left:5px solid var(--gold);\n    }\n\n    .result {\n        border-top:5px solid var(--gold);\n    }\n\n    .result-label {\n        color:var(--muted);\n        font-size:.82rem;\n        font-weight:700;\n        text-transform:uppercase;\n    }\n\n    .result-value {\n        color:var(--graphite);\n        font-size:clamp(1.5rem,4vw,1.9rem);\n        font-weight:800;\n    }\n\n    [data-baseweb="tab-list"] {\n        overflow-x:auto;\n        overflow-y:hidden;\n        white-space:nowrap;\n        gap:.2rem;\n    }\n\n    [data-baseweb="tab"] {\n        flex:0 0 auto;\n        min-width:max-content;\n    }\n\n    [data-testid="stFileUploaderDropzone"] {\n        background:var(--surface);\n        border:1.5px dashed #B8944E;\n        border-radius:14px;\n    }\n\n    [data-testid="stMetric"] {\n        background:var(--surface);\n        border:1px solid var(--border);\n        border-left:4px solid var(--gold);\n        border-radius:12px;\n        padding:.6rem .75rem;\n        min-width:0;\n    }\n\n    div.stButton>button {\n        background:var(--gold);\n        color:#171717;\n        border:1px solid #A96F08;\n        border-radius:12px;\n        font-weight:800;\n    }\n\n    div.stButton>button:hover {\n        background:var(--gold-soft);\n        color:#111;\n    }\n\n    /* Asegura que el texto del botón primario se vea en negrita. */\n    div.stButton>button p,\n    [data-testid="stBaseButton-primary"] p,\n    button[kind="primary"] p {\n        font-weight:800 !important;\n    }\n\n    [data-baseweb="select"] > div {\n        background:var(--surface);\n        border-color:#B8944E;\n    }\n\n    img,[data-testid="stImage"],[data-testid="stDataFrame"],\n    [data-testid="stVegaLiteChart"] {\n        max-width:100%;\n    }\n\n    .footer {\n        margin-top:2rem;\n        padding-top:1rem;\n        border-top:1px solid var(--border);\n        color:var(--muted);\n        text-align:center;\n        font-size:.84rem;\n    }\n\n    /* Glosario: lectura completa sin scroll ni recorte horizontal */\n    .glossary-list {\n        width:100%;\n        display:flex;\n        flex-direction:column;\n        gap:.65rem;\n        margin-top:.9rem;\n    }\n\n    .glossary-item {\n        width:100%;\n        box-sizing:border-box;\n        background:var(--surface);\n        border:1px solid var(--border);\n        border-left:4px solid var(--gold);\n        border-radius:10px;\n        padding:.8rem .9rem;\n        overflow:visible;\n    }\n\n    .glossary-term {\n        color:var(--graphite);\n        font-weight:700;\n        margin-bottom:.25rem;\n        overflow-wrap:anywhere;\n    }\n\n    .glossary-definition {\n        color:var(--graphite);\n        line-height:1.45;\n        white-space:normal;\n        overflow-wrap:anywhere;\n        word-break:normal;\n    }\n\n    @media(max-width:640px) {\n        .block-container {\n            padding:4.7rem .75rem 1.5rem .75rem;\n        }\n\n        [data-testid="stHorizontalBlock"] {\n            flex-wrap:wrap;\n            gap:.5rem;\n        }\n\n        [data-testid="column"] {\n            min-width:min(100%,210px)!important;\n            flex:1 1 210px!important;\n        }\n    }\n    </style>\n    ', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_classifier(model_path: str) -> Cifar10Classifier:
    """Carga una sola instancia del modelo durante la sesión."""
    classifier = Cifar10Classifier(Path(model_path))
    classifier.load()
    return classifier

@st.cache_data(show_spinner=False)
def load_test_manifest() -> pd.DataFrame:
    """Carga el índice de las 30 imágenes precargadas."""
    columns = ['filename', 'expected_label', 'source_filename']
    if not TEST_MANIFEST.exists():
        return pd.DataFrame(columns=columns)
    try:
        manifest = pd.read_csv(TEST_MANIFEST)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)
    if not {'filename', 'expected_label'}.issubset(manifest.columns):
        return pd.DataFrame(columns=columns)
    return manifest.sort_values('filename', key=lambda s: s.str.extract('(\\d+)', expand=False).astype(int)).reset_index(drop=True)

def get_model_path() -> tuple[Path | None, str]:
    """Prioriza el modelo final sobre el smoke."""
    if CONFIG.model_path.exists():
        return (CONFIG.model_path, 'final')
    if CONFIG.smoke_model_path.exists():
        return (CONFIG.smoke_model_path, 'smoke')
    return (None, 'missing')

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def load_csv(path: Path, index_col=None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, index_col=index_col)
        return frame if not frame.empty else None
    except (OSError, pd.errors.ParserError):
        return None

def make_fixed_thumbnail(image: Image.Image, size: tuple[int, int]=(320, 220)) -> Image.Image:
    """Crea una miniatura de tamaño fijo sin deformar la imagen."""
    rgb = image.convert('RGB')
    contained = ImageOps.contain(rgb, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, (255, 255, 255))
    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas

def render_header() -> None:
    left, right = st.columns([1, 4], vertical_alignment="center")
    with left:
        if CONFIG.logo_path.exists():
            st.image(str(CONFIG.logo_path), width=125)
    with right:
        st.markdown('<div class="kicker">Visión por computadora</div>', unsafe_allow_html=True)
        st.title("Eikonorasís CIFAR-10 akrivís")
        st.write(
            "Clasificación multiclase mediante Xception + Transfer Learning "
            "adaptado a CIFAR-10."
        )

def record_quick_test(filename: str, expected: str, predicted: str, confidence: float) -> None:
    """Acumula las pruebas rápidas de imágenes precargadas en la sesión."""
    if 'quick_test_results' not in st.session_state:
        st.session_state.quick_test_results = {}
    st.session_state.quick_test_results[filename] = {'Imagen': filename, 'Esperado': expected, 'Predicho': predicted, 'Confianza': confidence, 'Correcto': expected == predicted}

def record_uploaded_image(filename: str, predicted: str, confidence: float) -> None:
    """Registra temporalmente una clasificación hecha desde Subir imagen."""
    if 'uploaded_image_results' not in st.session_state:
        st.session_state.uploaded_image_results = []
    st.session_state.uploaded_image_results.append({'Prueba': len(st.session_state.uploaded_image_results) + 1, 'Imagen': filename, 'Predicho': predicted, 'Confianza': confidence})

def render_upload_history() -> None:
    """Muestra solo el registro temporal de imágenes subidas en la sesión."""
    results = st.session_state.get('uploaded_image_results', [])
    if not results:
        return
    frame = pd.DataFrame(results)
    display = frame.copy()
    display['Confianza'] = (display['Confianza'] * 100).map(lambda value: f'{value:.2f}%')
    st.markdown('#### Registro temporal de imágenes subidas')
    c1, c2 = st.columns(2)
    c1.metric('Imágenes analizadas', str(len(display)))
    c2.metric('Última predicción', str(display.iloc[-1]['Predicho']))
    st.dataframe(display, hide_index=True, width='stretch')
    st.caption('Este registro pertenece únicamente a la sección Subir imagen y es independiente de la Evaluación rápida de Imágenes precargadas.')
    if st.button('Limpiar registro de imágenes subidas', key='clear_uploaded_history'):
        st.session_state.uploaded_image_results = []
        st.rerun()

def render_quick_test_summary() -> None:
    results = st.session_state.get('quick_test_results', {})
    if not results:
        return
    frame = pd.DataFrame(results.values()).sort_values('Imagen')
    total = len(frame)
    correct = int(frame['Correcto'].sum())
    accuracy = correct / total if total else 0.0
    st.markdown('#### Evaluación rápida de esta sesión')
    c1, c2, c3 = st.columns(3)
    c1.metric('Imágenes probadas', str(total))
    c2.metric('Aciertos', f'{correct}/{total}')
    c3.metric('Accuracy rápida', f'{accuracy * 100:.2f}%')
    display = frame.copy()
    display['Confianza'] = (display['Confianza'] * 100).map(lambda value: f'{value:.2f}%')
    display['Correcto'] = display['Correcto'].map({True: 'Sí', False: 'No'})
    st.dataframe(display, hide_index=True, width='stretch')
    st.caption('Esta prueba rápida usa imágenes externas precargadas y no sustituye la evaluación oficial de CIFAR-10.')
    if st.button('Limpiar evaluación rápida', key='clear_quick_tests'):
        st.session_state.quick_test_results = {}
        st.session_state.confirmed_test_image = None
        st.rerun()

def select_preloaded_image() -> tuple[Image.Image | None, str | None, str | None]:
    """Explora 30 imágenes mediante un selector desplegable y confirma una selección."""
    manifest = load_test_manifest()
    if manifest.empty:
        st.warning('No se encontró el dataset visual precargado.')
        return (None, None, None)
    options = manifest['filename'].tolist()
    if 'explored_test_image' not in st.session_state:
        st.session_state.explored_test_image = options[0]
    if 'confirmed_test_image' not in st.session_state:
        st.session_state.confirmed_test_image = None
    st.markdown('#### Dataset visual de prueba')
    st.caption('Explora el archivo desde la lista. La miniatura se actualiza antes de confirmar la selección.')

    def option_label(filename: str) -> str:
        row = manifest.loc[manifest['filename'] == filename].iloc[0]
        stem = Path(filename).stem
        return f"{stem} — {row['expected_label']}"
    explored = st.selectbox('Explorar imagen precargada', options=options, index=options.index(st.session_state.explored_test_image), format_func=option_label, key='preloaded_selectbox')
    st.session_state.explored_test_image = explored
    row = manifest.loc[manifest['filename'] == explored].iloc[0]
    expected = str(row['expected_label'])
    image_path = TEST_IMAGES_DIR / explored
    if not image_path.exists():
        st.error(f'No se encontró el archivo {explored}.')
        return (None, None, None)
    with Image.open(image_path) as image:
        image.load()
        explored_image = image.convert('RGB')
    thumbnail = make_fixed_thumbnail(explored_image, size=(320, 220))
    preview_col, info_col = st.columns([1, 1.4], vertical_alignment='center')
    with preview_col:
        st.image(thumbnail, width=320)
    with info_col:
        st.markdown(f'\n            <div class="preview-box">\n            <strong>Archivo:</strong> {Path(explored).stem}<br>\n            <strong>Clase esperada:</strong> {expected}<br><br>\n            Esta es solo la previsualización. Confirma la imagen para habilitar\n            su clasificación.\n            </div>\n            ', unsafe_allow_html=True)
        if st.button('Seleccionar esta imagen', type='primary', width='stretch', key='confirm_preloaded_image'):
            st.session_state.confirmed_test_image = explored
            st.rerun()
    confirmed = st.session_state.get('confirmed_test_image')
    if confirmed is None:
        st.info('Confirma una imagen para continuar con la clasificación.')
        return (None, None, None)
    confirmed_rows = manifest.loc[manifest['filename'] == confirmed]
    if confirmed_rows.empty:
        st.session_state.confirmed_test_image = None
        return (None, None, None)
    confirmed_row = confirmed_rows.iloc[0]
    confirmed_expected = str(confirmed_row['expected_label'])
    confirmed_path = TEST_IMAGES_DIR / confirmed
    if not confirmed_path.exists():
        return (None, None, None)
    with Image.open(confirmed_path) as image:
        image.load()
        confirmed_image = image.convert('RGB')
    st.markdown(f'\n        <div class="status">\n        Imagen confirmada: <strong>{Path(confirmed).stem}</strong> ·\n        Clase esperada: <strong>{confirmed_expected}</strong>\n        </div>\n        ', unsafe_allow_html=True)
    return (confirmed_image, confirmed, confirmed_expected)

def render_prediction(model_path: Path, image: Image.Image, *, source_filename: str | None=None, expected_label: str | None=None, uploaded_filename: str | None=None) -> None:
    """Ejecuta inferencia y registra, si aplica, una prueba rápida."""
    try:
        predictions = get_classifier(str(model_path)).predict(image, top_k=3)
        best = predictions[0]
        st.markdown(f'\n            <div class="result">\n                <div class="result-label">Clase estimada</div>\n                <div class="result-value">{best.label}</div>\n                <div>\n                    Confianza:\n                    <strong>{best.probability * 100:.2f}%</strong>\n                </div>\n            </div>\n            ', unsafe_allow_html=True)
        if expected_label is not None:
            correct = best.label == expected_label
            verdict = 'Coincide' if correct else 'No coincide'
            symbol = '✅' if correct else '❌'
            st.write(f'**Clase esperada:** {expected_label} · **Resultado:** {symbol} {verdict}')
            if source_filename is not None:
                record_quick_test(source_filename, expected_label, best.label, best.probability)
        if uploaded_filename is not None:
            record_uploaded_image(uploaded_filename, best.label, best.probability)
        st.markdown('#### Top-3')
        for prediction in predictions:
            st.write(f'**{prediction.label}** — {prediction.probability * 100:.2f}%')
            st.progress(prediction.probability)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.exception('Error de inferencia: %s', exc)
        st.error('No fue posible completar la inferencia.')

def render_classifier(model_path: Path | None) -> None:
    st.subheader('Clasificar imagen')
    with st.expander('¿Qué puede reconocer el modelo?'):
        st.write(', '.join(CLASS_NAMES_ES))
    source = st.radio('Fuente de imagen', ['Imágenes precargadas', 'Subir imagen'], horizontal=True)
    if source == 'Imágenes precargadas':
        image, filename, expected = select_preloaded_image()
        if image is not None and st.button('Analizar imagen seleccionada', type='primary', width='stretch', disabled=model_path is None, key='analyze_preloaded'):
            render_prediction(model_path, image, source_filename=filename, expected_label=expected)
        render_quick_test_summary()
        return
    st.caption('JPG, JPEG o PNG · Máximo 5 MB')
    uploaded = st.file_uploader('Selecciona una imagen', type=['jpg', 'jpeg', 'png'])
    if uploaded is None:
        st.markdown('<div class="status status-dark">Carga una imagen para iniciar el análisis.</div>', unsafe_allow_html=True)
        render_upload_history()
        return
    try:
        image = validate_uploaded_image(content=uploaded.getvalue(), filename=uploaded.name, allowed_extensions=CONFIG.allowed_extensions, max_bytes=CONFIG.max_upload_bytes)
    except ImageValidationError as exc:
        st.error(str(exc))
        render_upload_history()
        return
    upload_preview = make_fixed_thumbnail(image, size=(480, 320))
    preview_left, preview_center, preview_right = st.columns([1, 2, 1])
    with preview_center:
        st.image(upload_preview, caption='Imagen seleccionada', width=480)
    if st.button('Analizar imagen', type='primary', width='stretch', disabled=model_path is None, key='analyze_upload'):
        render_prediction(model_path, image, uploaded_filename=uploaded.name)
    render_upload_history()

def render_dataset() -> None:
    st.subheader('Dataset CIFAR-10')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Imágenes', '60,000')
    c2.metric('Resolución', '32×32')
    c3.metric('Canales', 'RGB')
    c4.metric('Clases', '10')
    st.dataframe(pd.DataFrame({'Partición': ['Entrenamiento', 'Validación', 'Prueba oficial'], 'Imágenes': [45000, 5000, 10000], 'Propósito': ['Aprendizaje de parámetros', 'Control de generalización', 'Evaluación final']}), hide_index=True, width='stretch')
    st.write('**Clases:** ' + ', '.join(CLASS_NAMES_ES))
    st.markdown('### Dataset visual adicional')
    st.write('La aplicación incluye **30 imágenes externas precargadas**, tres por categoría, renombradas de `img_test1` a `img_test30`. Se utilizan únicamente para pruebas rápidas de inferencia y experiencia de usuario.')
    st.caption('Estas 30 imágenes no se mezclan con las 10,000 imágenes oficiales utilizadas para las métricas finales.')
    st.markdown('### Fuentes y crédito del dataset')
    st.markdown('\n        **Fuente académica principal:**  \n        [UCI Machine Learning Repository — CIFAR-10](https://archive.ics.uci.edu/dataset/691/cifar+10)\n\n        UCI documenta CIFAR-10 como un dataset de **60,000 imágenes a color\n        de 32×32 píxeles**, organizadas en **10 clases**, con una partición\n        estándar de **50,000 imágenes de entrenamiento** y **10,000 de prueba**.\n\n        **Referencia formal sugerida:**  \n        *CIFAR-10 [Dataset]. (2009). UCI Machine Learning Repository.*\n        DOI: [10.24432/C5889J](https://doi.org/10.24432/C5889J)\n\n        **Referencia práctica complementaria:**  \n        [Kaggle — CIFAR-10 Object Recognition in Images](https://www.kaggle.com/c/cifar-10)\n\n        Kaggle se utiliza como referencia complementaria para consultar el\n        problema de reconocimiento de objetos y su contexto práctico.\n        ')

def render_model() -> None:
    st.subheader("Modelo evaluador")
    st.markdown(
        "**Xception + Transfer Learning** con ajuste fino controlado para "
        "clasificación multiclase de CIFAR-10."
    )
    st.markdown("### Flujo del evaluador")

    st.image(str(PROJECT_ROOT / "assets" / "flujo_clasificacion_xception.png"), width="stretch")
    st.markdown("Flujo: imagen → preprocesamiento → Xception → Transfer Learning → fine-tuning → clasificación")
    st.caption("CIFAR-10 · 10 categorías · inferencia con model_final.keras")

    st.table({
        "Parámetro": [
            "Modelo", "Arquitectura base", "Pesos iniciales", "Entrada",
            "Salida", "Clasificador", "Top-k mostrado"
        ],
        "Valor": [
            "model_final.keras", "Xception", "ImageNet", "RGB 160×160×3",
            "10 clases", "Dense(10) + Softmax", "3"
        ],
    })
    st.info(
        "Esta interfaz utiliza el modelo final persistido para inferencia. "
        "No se ejecuta entrenamiento desde la aplicación."
    )

def render_training_evaluation() -> None:
    st.subheader("Entrenamiento y evaluación")
    st.markdown("### Estado del evaluador final")
    st.write(
        "El evaluador utilizado por la aplicación es **model_final.keras**, "
        "basado en Xception + Transfer Learning. En esta etapa no se realiza "
        "una nueva optimización de precisión."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Test CIFAR-10", "90.36%")
    c2.metric("Clases", "10")
    c3.metric("Imágenes de prueba", "10,000")
    st.markdown("### Prueba funcional de las 30 imágenes")
    st.write(
        "Las 30 imágenes precargadas se conservan como prueba funcional de "
        "la interfaz. No sustituyen la evaluación oficial."
    )
    st.warning(
        " como si pertenecieran "
        "al evaluador Xception."
    )

def render_about() -> None:
    st.subheader("Acerca de Eikonorasís")
    st.write(
        "**Eikonorasís CIFAR-10 akrivís** es una aplicación académica de "
        "visión por computadora para clasificación multiclase de imágenes."
    )
    st.write(
        "El modelo evaluador utiliza Xception + Transfer Learning y "
        "preprocesamiento RGB a 160×160."
    )
    st.write(
        "El dominio está limitado a las diez categorías oficiales de CIFAR-10."
    )
    st.write(
        "**Privacidad:** las imágenes cargadas se procesan para inferencia "
        "y no se requiere almacenamiento permanente de la imagen."
    )

def render_trivia() -> None:
    st.subheader("Trivia: ¿de dónde viene Eikonorasís?")
    st.markdown(
        """
**Eikonorasís** es un neologismo creado para esta aplicación e inspirado
en dos raíces griegas:

- **Eikón (εἰκών):** imagen, figura o representación.
- **Órasis (ὅρασις):** visión o acto de ver.

El nombre puede entenderse libremente como **“visión de imágenes”**,
**“visión icónica”** o **“iconovisión”**.

### ¿Qué significa *akrivís*?

**Akrivís** es un término de origen griego relacionado con **precisión y exactitud**. En **Eikonorasís CIFAR-10 akrivís**, identifica la versión final del proyecto y expresa su enfoque en una clasificación **precisa, medible y evaluable**.

**Idea central:** *Eikonorasís* representa la visión; *CIFAR-10*, las diez categorías; y *akrivís*, el principio de precisión que distingue la versión final.

### ¿Qué representa el logo?

- **El ojo:** percepción y visión.
- **Los píxeles:** representación digital de la imagen.
- **Los nodos conectados:** red neuronal y aprendizaje profundo.
- **El dorado y grafito:** identidad visual de Eikonorasís.
- **CIFAR-10:** dataset compuesto por diez categorías.
"""
    )

def render_help() -> None:
    st.subheader("Ayuda")
    help_tab1, help_tab2, help_tab3 = st.tabs(
        ["Cómo funciona", "Tecnologías", "Glosario"]
    )
    with help_tab1:
        st.markdown(
            """
### Inicio rápido

1. Abre **Clasificar**.
2. Elige **Imágenes precargadas** o **Subir imagen**.
3. Selecciona o carga una imagen.
4. Revisa la previsualización y confirma cuando corresponda.
5. Ejecuta el análisis.
6. Interpreta **clase estimada**, **confianza** y **Top-3**.

### Imágenes precargadas

Selecciona `img_test1` a `img_test30`, revisa la miniatura y la clase
esperada, confirma la imagen y ejecuta el análisis.

- **Clase esperada:** categoría conocida de la imagen de prueba.
- **Clase estimada:** categoría predicha por el evaluador.
- **Confianza:** probabilidad de la predicción principal.
- **Top-3:** tres clases con mayor probabilidad.
- **Evaluación rápida:** comparación esperado vs. predicho durante la sesión.

Esta prueba es funcional y no sustituye la evaluación oficial de CIFAR-10.

### Imagen propia

Carga un JPG, JPEG o PNG válido. La aplicación valida el archivo, lo
convierte a RGB y lo prepara a **160×160** antes de la inferencia.

### Confianza

**Confianza no significa certeza.** Softmax distribuye la probabilidad entre
las diez clases. Una imagen externa puede estar fuera de la distribución.

La app reconoce: **avión, automóvil, ave, gato, ciervo, perro, rana,
caballo, barco y camión**.
"""
        )
    with help_tab2:
        st.markdown(
            """
### Tecnologías

| Capa | Tecnología | Función |
|---|---|---|
| Interfaz | Streamlit | Navegación y resultados |
| Validación | Python + Pillow | Validación de imágenes |
| Preprocesamiento | Pillow + NumPy | RGB y 160×160 |
| Modelo | TensorFlow / Keras | Inferencia |
| Arquitectura | Xception | Extracción de características |
| Transfer Learning | ImageNet | Representaciones iniciales |
| Fine-tuning | Keras | Adaptación controlada |
| Clasificación | Dense(10) + Softmax | Diez probabilidades |

**Flujo:** Usuario → Streamlit → validación → RGB 160×160 → Xception →
Transfer Learning → fine-tuning → Softmax → predicción.
"""
        )
    with help_tab3:
        st.markdown(
            """
### Glosario

- **CIFAR-10:** conjunto de diez categorías de imágenes.
- **CNN:** red neuronal convolucional para patrones visuales.
- **Xception:** arquitectura convolucional usada como modelo base.
- **Transfer Learning:** reutilización de representaciones aprendidas.
- **Fine-tuning:** ajuste controlado del modelo para el dominio objetivo.
- **Inferencia:** ejecución del modelo sobre una imagen nueva.
- **Softmax:** convierte las salidas en probabilidades normalizadas.
- **Confianza:** probabilidad asociada a la clase predicha.
- **Top-3:** tres clases con mayor probabilidad.
- **Preprocesamiento:** transformación previa a la inferencia.
"""
        )

def render_institutional() -> None:
    """Identificación académica e institucional del proyecto."""
    st.subheader('Institucional')
    st.caption('Identificación académica de Eikonorasís CIFAR-10 akrivís y del contexto formativo en el que fue desarrollado.')
    st.markdown('\n        <div class="analysis-box">\n        <strong style="font-size:.82rem;letter-spacing:.05em;">PROYECTO ACADÉMICO</strong><br><br>\n        <strong style="font-size:1.35rem;">INSTITUTO INTERNACIONAL DE AGUASCALIENTES</strong><br>\n        Maestría en Inteligencia Artificial para la Transformación Digital\n        </div>\n        ', unsafe_allow_html=True)
    st.link_button('Sitio oficial del Instituto Internacional de Aguascalientes', 'https://www.iinternacional.edu.mx/')
    st.markdown('### Información académica')
    st.markdown('\n        **Asignatura:** Aprendizaje Profundo  \n        **Aplicación:** Eikonorasís CIFAR-10 akrivís  \n        **Proyecto:** Diseño, implementación, entrenamiento, evaluación y\n        despliegue web de una red neuronal convolucional en Python para la\n        clasificación multiclase de imágenes mediante CIFAR-10.\n        ')
    st.markdown('### Autoría académica')
    st.markdown('\n        **Alumno:** Antonio Nicolás Toro González  \n        **Tutora:** Dra. Claudia Andrea Vidales Basurto\n        ')
    st.markdown('### Descripción del proyecto')
    st.write('Eikonorasís CIFAR-10 akrivís es una aplicación web interactiva desarrollada en Python que utiliza una Xception + Transfer Learning para clasificar imágenes en las diez categorías de CIFAR-10. El proyecto integra preparación y partición de datos, preprocesamiento, data augmentation, diseño de la arquitectura, entrenamiento, validación, evaluación sobre un conjunto de prueba independiente, análisis de errores, inferencia y despliegue web.')
    st.markdown('### Identificación técnica')
    institutional_tech = pd.DataFrame({'Componente': ['Lenguaje', 'Modelo', 'Deep Learning', 'Preprocesamiento', 'Evaluación', 'Interfaz web', 'Dataset', 'Repositorio', 'Aplicación web'], 'Tecnología / referencia': ['Python 3.11', 'Xception + Transfer Learning para clasificación multiclase', 'TensorFlow / Keras + Keras Applications', 'Pillow + NumPy', 'scikit-learn + pandas', 'Streamlit', 'CIFAR-10 — UCI Machine Learning Repository', 'https://github.com/edtech-mx-ve/eikonorasis-cifar10', 'https://eikonorasis-cifar10.streamlit.app/']})
    st.dataframe(institutional_tech, hide_index=True, width='stretch')

def main() -> None:
    render_header()
    model_path, mode = get_model_path()
    if mode == "final":
        st.markdown(
            '<div class="status">Modelo final disponible para inferencia.</div>',
            unsafe_allow_html=True,
        )
    elif mode == "smoke":
        st.markdown(
            '<div class="status status-warning">Modo demostración: se está usando el modelo smoke.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("No se encontró un modelo entrenado.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Clasificar", "Dataset", "Modelo", "Entrenamiento y evaluación",
            "Acerca de", "Trivia", "Ayuda", "Institucional"
        ]
    )
    with tab1:
        render_classifier(model_path)
    with tab2:
        render_dataset()
    with tab3:
        render_model()
    with tab4:
        render_training_evaluation()
    with tab5:
        render_about()
    with tab6:
        render_trivia()
    with tab7:
        render_help()
    with tab8:
        render_institutional()
    st.markdown(
        '<div class="footer">Eikonorasís CIFAR-10 akrivís · Proyecto académico de aprendizaje profundo</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

