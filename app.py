"""EikonorasÃ­s CIFAR-10 akrivÃ­s â€” clasificaciÃ³n, evaluaciÃ³n y selector desplegable de imÃ¡genes."""
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

# Recurso grÃ¡fico del modelo: ruta absoluta basada en la raÃ­z de la app.
DIAGRAM_PATH = Path(__file__).resolve().parent / "assets" / "modelo_xception_eikonorasis.svg"

TRIVIA_TEXT = '### Â¿De dÃ³nde viene EikonorasÃ­s?\n\n**EikonorasÃ­s** es un neologismo creado para esta aplicaciÃ³n e inspirado en dos raÃ­ces griegas:\n\n- **EikÃ³n (Îµá¼°ÎºÏŽÎ½):** imagen, figura o representaciÃ³n.\n- **Ã“rasis (á½…ÏÎ±ÏƒÎ¹Ï‚):** visiÃ³n o acto de ver.\n\nEl nombre sintetiza la funciÃ³n de la aplicaciÃ³n: recibir una imagen, procesarla mediante un modelo de visiÃ³n por computadora y producir una clasificaciÃ³n visual.\n\n### Â¿QuÃ© significa *akrivÃ­s*?\n\n**AkrivÃ­s (á¼€ÎºÏÎ¹Î²Î®Ï‚)** alude a precisiÃ³n, exactitud o rigurosidad. En el proyecto, **EikonorasÃ­s CIFAR-10 akrivÃ­s** identifica la versiÃ³n final del sistema y su evaluador de clasificaciÃ³n.\n\n### Â¿QuÃ© representa el logo?\n\n- **El ojo:** percepciÃ³n y visiÃ³n.\n- **Los pÃ­xeles:** la imagen digital de entrada.\n- **Los nodos conectados:** red neuronal y aprendizaje profundo.\n- **Dorado y grafito:** identidad visual de EikonorasÃ­s.\n\n**CIFAR-10** identifica el conjunto de datos utilizado por el proyecto, con diez categorÃ­as oficiales.\n\n**Idea central:** EikonorasÃ­s transforma una imagen digital en una decisiÃ³n de clasificaciÃ³n mediante visiÃ³n por computadora.\n'
HELP_TEXT = '### TecnologÃ­a\n\nEl evaluador final utiliza **Xception + Transfer Learning** y ajuste fino (*fine-tuning*) para clasificar las diez categorÃ­as de CIFAR-10.\n\n**Flujo de inferencia**\n\nImagen â†’ RGB â†’ 160Ã—160 â†’ Xception â†’ Transfer Learning â†’ fine-tuning â†’ 10 clases â†’ predicciÃ³n y confianza.\n\n### Glosario\n\n- **Xception:** arquitectura convolucional utilizada como modelo base preentrenado.\n- **Transfer Learning:** reutilizaciÃ³n de representaciones aprendidas previamente para un nuevo problema.\n- **Fine-tuning:** ajuste controlado de parte del modelo con una tasa de aprendizaje menor.\n- **Inferencia:** ejecuciÃ³n del modelo entrenado sobre una imagen nueva.\n- **Top-3:** tres clases con mayor probabilidad predicha.\n- **Confianza:** probabilidad asociada a la clase predicha.\n'

TRAINING_TEXT = '### Resultados finales del modelo evaluador\n\nEl evaluador final utiliza **Xception + Transfer Learning**, con ajuste fino controlado, para la clasificaciÃ³n de las 10 clases oficiales de CIFAR-10.\n\n**EvaluaciÃ³n formal del modelo:**\n- ValidaciÃ³n: **91.10 %**\n- Test CIFAR-10: **90.36 %**\n- Modelo promovido: `model_final.keras`\n\n**Prueba funcional de la aplicaciÃ³n:**\n- ImÃ¡genes precargadas procesadas: **30/30**\n- Aciertos: **30/30**\n- Accuracy funcional demostrativa: **100.00 %**\n\nLa prueba de 30 imÃ¡genes es funcional y demostrativa; no sustituye la evaluaciÃ³n formal sobre el conjunto de prueba de CIFAR-10.\n\nNo se realizÃ³ entrenamiento adicional durante la integraciÃ³n de la aplicaciÃ³n.\n'

MODEL_TEXT = '### Modelo evaluador\n\n**Xception + Transfer Learning** sobre CIFAR-10.\n\n- **Modelo base:** Xception preentrenada con pesos de ImageNet.\n- **Entrada de inferencia:** imagen RGB redimensionada a **160Ã—160Ã—3**.\n- **Transfer Learning:** extracciÃ³n de representaciones visuales a partir del modelo preentrenado.\n- **Fine-tuning:** ajuste controlado para adaptar el evaluador al problema de 10 clases.\n- **Salida:** 10 categorÃ­as oficiales de CIFAR-10.\n- **Inferencia:** clase predicha, confianza y Top-3.\n\nEl modelo utilizado por la aplicaciÃ³n es `model_final.keras`.\n'

DIAGRAM = PROJECT_ROOT / "assets" / "modelo_xception_transfer_learning.svg"
configure_logging()
LOGGER = logging.getLogger(__name__)
CONFIG = AppConfig()
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODEL_DIR = PROJECT_ROOT / 'model'
TEST_IMAGES_DIR = PROJECT_ROOT / 'assets' / 'test_images'
TEST_MANIFEST = TEST_IMAGES_DIR / 'manifest.csv'
st.set_page_config(page_title='EikonorasÃ­s CIFAR-10 akrivÃ­s', page_icon='ðŸ‘ï¸', layout='centered')
st.markdown('\n    <style>\n    :root {\n        --bg:#F7F4EE; --surface:#FFFFFF; --graphite:#23262B;\n        --gold:#C88A12; --gold-soft:#E7BF67; --gold-pale:#F8EED6;\n        --border:#DED8CC; --muted:#62666D;\n    }\n\n    html, body { overflow-x:hidden; }\n\n    .stApp {\n        background:var(--bg);\n        color:var(--graphite);\n    }\n\n    .block-container {\n        width:min(100%,1020px);\n        max-width:1020px;\n        margin:0 auto;\n        padding:5.2rem 1.2rem 2rem 1.2rem;\n    }\n\n    h1,h2,h3 {\n        color:var(--graphite);\n        overflow-wrap:anywhere;\n    }\n\n    h1 {\n        font-size:clamp(2rem,4vw,3rem);\n        line-height:1.08;\n    }\n\n    .kicker {\n        color:#7B570B;\n        font-size:.78rem;\n        font-weight:800;\n        letter-spacing:.08em;\n        text-transform:uppercase;\n    }\n\n    .status,.analysis-box,.result,.preview-box {\n        background:var(--surface);\n        border:1px solid var(--border);\n        border-radius:14px;\n        padding:.9rem 1rem;\n        margin:.8rem 0;\n    }\n\n    .status {\n        background:var(--gold-pale);\n        border-left:6px solid var(--gold);\n    }\n\n    .status-warning {\n        background:#FFF5E5;\n        border-left-color:#A66B00;\n    }\n\n    .status-dark {\n        background:#EFF0F1;\n        border-left-color:var(--graphite);\n    }\n\n    .analysis-box {\n        background:#FFFDF8;\n        border-left:5px solid var(--gold);\n    }\n\n    .result {\n        border-top:5px solid var(--gold);\n    }\n\n    .result-label {\n        color:var(--muted);\n        font-size:.82rem;\n        font-weight:700;\n        text-transform:uppercase;\n    }\n\n    .result-value {\n        color:var(--graphite);\n        font-size:clamp(1.5rem,4vw,1.9rem);\n        font-weight:800;\n    }\n\n    [data-baseweb="tab-list"] {\n        overflow-x:auto;\n        overflow-y:hidden;\n        white-space:nowrap;\n        gap:.2rem;\n    }\n\n    [data-baseweb="tab"] {\n        flex:0 0 auto;\n        min-width:max-content;\n    }\n\n    [data-testid="stFileUploaderDropzone"] {\n        background:var(--surface);\n        border:1.5px dashed #B8944E;\n        border-radius:14px;\n    }\n\n    [data-testid="stMetric"] {\n        background:var(--surface);\n        border:1px solid var(--border);\n        border-left:4px solid var(--gold);\n        border-radius:12px;\n        padding:.6rem .75rem;\n        min-width:0;\n    }\n\n    div.stButton>button {\n        background:var(--gold);\n        color:#171717;\n        border:1px solid #A96F08;\n        border-radius:12px;\n        font-weight:800;\n    }\n\n    div.stButton>button:hover {\n        background:var(--gold-soft);\n        color:#111;\n    }\n\n    /* Asegura que el texto del botÃ³n primario se vea en negrita. */\n    div.stButton>button p,\n    [data-testid="stBaseButton-primary"] p,\n    button[kind="primary"] p {\n        font-weight:800 !important;\n    }\n\n    [data-baseweb="select"] > div {\n        background:var(--surface);\n        border-color:#B8944E;\n    }\n\n    img,[data-testid="stImage"],[data-testid="stDataFrame"],\n    [data-testid="stVegaLiteChart"] {\n        max-width:100%;\n    }\n\n    .footer {\n        margin-top:2rem;\n        padding-top:1rem;\n        border-top:1px solid var(--border);\n        color:var(--muted);\n        text-align:center;\n        font-size:.84rem;\n    }\n\n    /* Glosario: lectura completa sin scroll ni recorte horizontal */\n    .glossary-list {\n        width:100%;\n        display:flex;\n        flex-direction:column;\n        gap:.65rem;\n        margin-top:.9rem;\n    }\n\n    .glossary-item {\n        width:100%;\n        box-sizing:border-box;\n        background:var(--surface);\n        border:1px solid var(--border);\n        border-left:4px solid var(--gold);\n        border-radius:10px;\n        padding:.8rem .9rem;\n        overflow:visible;\n    }\n\n    .glossary-term {\n        color:var(--graphite);\n        font-weight:700;\n        margin-bottom:.25rem;\n        overflow-wrap:anywhere;\n    }\n\n    .glossary-definition {\n        color:var(--graphite);\n        line-height:1.45;\n        white-space:normal;\n        overflow-wrap:anywhere;\n        word-break:normal;\n    }\n\n    @media(max-width:640px) {\n        .block-container {\n            padding:4.7rem .75rem 1.5rem .75rem;\n        }\n\n        [data-testid="stHorizontalBlock"] {\n            flex-wrap:wrap;\n            gap:.5rem;\n        }\n\n        [data-testid="column"] {\n            min-width:min(100%,210px)!important;\n            flex:1 1 210px!important;\n        }\n    }\n    </style>\n    ', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_classifier(model_path: str) -> Cifar10Classifier:
    """Carga una sola instancia del modelo durante la sesiÃ³n."""
    classifier = Cifar10Classifier(Path(model_path))
    classifier.load()
    return classifier

@st.cache_data(show_spinner=False)
def load_test_manifest() -> pd.DataFrame:
    """Carga el Ã­ndice de las 30 imÃ¡genes precargadas."""
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
    """Crea una miniatura de tamaÃ±o fijo sin deformar la imagen."""
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
        st.markdown('<div class="kicker">VisiÃ³n por computadora</div>', unsafe_allow_html=True)
        st.title("EikonorasÃ­s CIFAR-10 akrivÃ­s")
        st.write(
            "ClasificaciÃ³n multiclase mediante Xception + Transfer Learning "
            "adaptado a CIFAR-10."
        )

def record_quick_test(filename: str, expected: str, predicted: str, confidence: float) -> None:
    """Acumula las pruebas rÃ¡pidas de imÃ¡genes precargadas en la sesiÃ³n."""
    if 'quick_test_results' not in st.session_state:
        st.session_state.quick_test_results = {}
    st.session_state.quick_test_results[filename] = {'Imagen': filename, 'Esperado': expected, 'Predicho': predicted, 'Confianza': confidence, 'Correcto': expected == predicted}

def record_uploaded_image(filename: str, predicted: str, confidence: float) -> None:
    """Registra temporalmente una clasificaciÃ³n hecha desde Subir imagen."""
    if 'uploaded_image_results' not in st.session_state:
        st.session_state.uploaded_image_results = []
    st.session_state.uploaded_image_results.append({'Prueba': len(st.session_state.uploaded_image_results) + 1, 'Imagen': filename, 'Predicho': predicted, 'Confianza': confidence})

def render_upload_history() -> None:
    """Muestra solo el registro temporal de imÃ¡genes subidas en la sesiÃ³n."""
    results = st.session_state.get('uploaded_image_results', [])
    if not results:
        return
    frame = pd.DataFrame(results)
    display = frame.copy()
    display['Confianza'] = (display['Confianza'] * 100).map(lambda value: f'{value:.2f}%')
    st.markdown('#### Registro temporal de imÃ¡genes subidas')
    c1, c2 = st.columns(2)
    c1.metric('ImÃ¡genes analizadas', str(len(display)))
    c2.metric('Ãšltima predicciÃ³n', str(display.iloc[-1]['Predicho']))
    st.dataframe(display, hide_index=True, width='stretch')
    st.caption('Este registro pertenece Ãºnicamente a la secciÃ³n Subir imagen y es independiente de la EvaluaciÃ³n rÃ¡pida de ImÃ¡genes precargadas.')
    if st.button('Limpiar registro de imÃ¡genes subidas', key='clear_uploaded_history'):
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
    st.markdown('#### EvaluaciÃ³n rÃ¡pida de esta sesiÃ³n')
    c1, c2, c3 = st.columns(3)
    c1.metric('ImÃ¡genes probadas', str(total))
    c2.metric('Aciertos', f'{correct}/{total}')
    c3.metric('Accuracy rÃ¡pida', f'{accuracy * 100:.2f}%')
    display = frame.copy()
    display['Confianza'] = (display['Confianza'] * 100).map(lambda value: f'{value:.2f}%')
    display['Correcto'] = display['Correcto'].map({True: 'SÃ­', False: 'No'})
    st.dataframe(display, hide_index=True, width='stretch')
    st.caption('Esta prueba rÃ¡pida usa imÃ¡genes externas precargadas y no sustituye la evaluaciÃ³n oficial de CIFAR-10.')
    if st.button('Limpiar evaluaciÃ³n rÃ¡pida', key='clear_quick_tests'):
        st.session_state.quick_test_results = {}
        st.session_state.confirmed_test_image = None
        st.rerun()

def select_preloaded_image() -> tuple[Image.Image | None, str | None, str | None]:
    """Explora 30 imÃ¡genes mediante un selector desplegable y confirma una selecciÃ³n."""
    manifest = load_test_manifest()
    if manifest.empty:
        st.warning('No se encontrÃ³ el dataset visual precargado.')
        return (None, None, None)
    options = manifest['filename'].tolist()
    if 'explored_test_image' not in st.session_state:
        st.session_state.explored_test_image = options[0]
    if 'confirmed_test_image' not in st.session_state:
        st.session_state.confirmed_test_image = None
    st.markdown('#### Dataset visual de prueba')
    st.caption('Explora el archivo desde la lista. La miniatura se actualiza antes de confirmar la selecciÃ³n.')

    def option_label(filename: str) -> str:
        row = manifest.loc[manifest['filename'] == filename].iloc[0]
        stem = Path(filename).stem
        return f"{stem} â€” {row['expected_label']}"
    explored = st.selectbox('Explorar imagen precargada', options=options, index=options.index(st.session_state.explored_test_image), format_func=option_label, key='preloaded_selectbox')
    st.session_state.explored_test_image = explored
    row = manifest.loc[manifest['filename'] == explored].iloc[0]
    expected = str(row['expected_label'])
    image_path = TEST_IMAGES_DIR / explored
    if not image_path.exists():
        st.error(f'No se encontrÃ³ el archivo {explored}.')
        return (None, None, None)
    with Image.open(image_path) as image:
        image.load()
        explored_image = image.convert('RGB')
    thumbnail = make_fixed_thumbnail(explored_image, size=(320, 220))
    preview_col, info_col = st.columns([1, 1.4], vertical_alignment='center')
    with preview_col:
        st.image(thumbnail, width=320)
    with info_col:
        st.markdown(f'\n            <div class="preview-box">\n            <strong>Archivo:</strong> {Path(explored).stem}<br>\n            <strong>Clase esperada:</strong> {expected}<br><br>\n            Esta es solo la previsualizaciÃ³n. Confirma la imagen para habilitar\n            su clasificaciÃ³n.\n            </div>\n            ', unsafe_allow_html=True)
        if st.button('Seleccionar esta imagen', type='primary', width='stretch', key='confirm_preloaded_image'):
            st.session_state.confirmed_test_image = explored
            st.rerun()
    confirmed = st.session_state.get('confirmed_test_image')
    if confirmed is None:
        st.info('Confirma una imagen para continuar con la clasificaciÃ³n.')
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
    st.markdown(f'\n        <div class="status">\n        Imagen confirmada: <strong>{Path(confirmed).stem}</strong> Â·\n        Clase esperada: <strong>{confirmed_expected}</strong>\n        </div>\n        ', unsafe_allow_html=True)
    return (confirmed_image, confirmed, confirmed_expected)

def render_prediction(model_path: Path, image: Image.Image, *, source_filename: str | None=None, expected_label: str | None=None, uploaded_filename: str | None=None) -> None:
    """Ejecuta inferencia y registra, si aplica, una prueba rÃ¡pida."""
    try:
        predictions = get_classifier(str(model_path)).predict(image, top_k=3)
        best = predictions[0]
        st.markdown(f'\n            <div class="result">\n                <div class="result-label">Clase estimada</div>\n                <div class="result-value">{best.label}</div>\n                <div>\n                    Confianza:\n                    <strong>{best.probability * 100:.2f}%</strong>\n                </div>\n            </div>\n            ', unsafe_allow_html=True)
        if expected_label is not None:
            correct = best.label == expected_label
            verdict = 'Coincide' if correct else 'No coincide'
            symbol = 'âœ…' if correct else 'âŒ'
            st.write(f'**Clase esperada:** {expected_label} Â· **Resultado:** {symbol} {verdict}')
            if source_filename is not None:
                record_quick_test(source_filename, expected_label, best.label, best.probability)
        if uploaded_filename is not None:
            record_uploaded_image(uploaded_filename, best.label, best.probability)
        st.markdown('#### Top-3')
        for prediction in predictions:
            st.write(f'**{prediction.label}** â€” {prediction.probability * 100:.2f}%')
            st.progress(prediction.probability)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.exception('Error de inferencia: %s', exc)
        st.error('No fue posible completar la inferencia.')

def render_classifier(model_path: Path | None) -> None:
    st.subheader('Clasificar imagen')
    with st.expander('Â¿QuÃ© puede reconocer el modelo?'):
        st.write(', '.join(CLASS_NAMES_ES))
    source = st.radio('Fuente de imagen', ['ImÃ¡genes precargadas', 'Subir imagen'], horizontal=True)
    if source == 'ImÃ¡genes precargadas':
        image, filename, expected = select_preloaded_image()
        if image is not None and st.button('Analizar imagen seleccionada', type='primary', width='stretch', disabled=model_path is None, key='analyze_preloaded'):
            render_prediction(model_path, image, source_filename=filename, expected_label=expected)
        render_quick_test_summary()
        return
    st.caption('JPG, JPEG o PNG Â· MÃ¡ximo 5 MB')
    uploaded = st.file_uploader('Selecciona una imagen', type=['jpg', 'jpeg', 'png'])
    if uploaded is None:
        st.markdown('<div class="status status-dark">Carga una imagen para iniciar el anÃ¡lisis.</div>', unsafe_allow_html=True)
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
    c1.metric('ImÃ¡genes', '60,000')
    c2.metric('ResoluciÃ³n', '32Ã—32')
    c3.metric('Canales', 'RGB')
    c4.metric('Clases', '10')
    st.dataframe(pd.DataFrame({'ParticiÃ³n': ['Entrenamiento', 'ValidaciÃ³n', 'Prueba oficial'], 'ImÃ¡genes': [45000, 5000, 10000], 'PropÃ³sito': ['Aprendizaje de parÃ¡metros', 'Control de generalizaciÃ³n', 'EvaluaciÃ³n final']}), hide_index=True, width='stretch')
    st.write('**Clases:** ' + ', '.join(CLASS_NAMES_ES))
    st.markdown('### Dataset visual adicional')
    st.write('La aplicaciÃ³n incluye **30 imÃ¡genes externas precargadas**, tres por categorÃ­a, renombradas de `img_test1` a `img_test30`. Se utilizan Ãºnicamente para pruebas rÃ¡pidas de inferencia y experiencia de usuario.')
    st.caption('Estas 30 imÃ¡genes no se mezclan con las 10,000 imÃ¡genes oficiales utilizadas para las mÃ©tricas finales.')
    st.markdown('### Fuentes y crÃ©dito del dataset')
    st.markdown('\n        **Fuente acadÃ©mica principal:**  \n        [UCI Machine Learning Repository â€” CIFAR-10](https://archive.ics.uci.edu/dataset/691/cifar+10)\n\n        UCI documenta CIFAR-10 como un dataset de **60,000 imÃ¡genes a color\n        de 32Ã—32 pÃ­xeles**, organizadas en **10 clases**, con una particiÃ³n\n        estÃ¡ndar de **50,000 imÃ¡genes de entrenamiento** y **10,000 de prueba**.\n\n        **Referencia formal sugerida:**  \n        *CIFAR-10 [Dataset]. (2009). UCI Machine Learning Repository.*\n        DOI: [10.24432/C5889J](https://doi.org/10.24432/C5889J)\n\n        **Referencia prÃ¡ctica complementaria:**  \n        [Kaggle â€” CIFAR-10 Object Recognition in Images](https://www.kaggle.com/c/cifar-10)\n\n        Kaggle se utiliza como referencia complementaria para consultar el\n        problema de reconocimiento de objetos y su contexto prÃ¡ctico.\n        ')

def render_model() -> None:
    st.subheader("Modelo evaluador")
    st.markdown(
        "**Xception + Transfer Learning** con ajuste fino controlado para "
        "clasificaciÃ³n multiclase de CIFAR-10."
    )
    st.markdown("### Flujo del evaluador")

    st.image(str(PROJECT_ROOT / "assets" / "flujo_clasificacion_xception.png"), width="stretch")
    st.markdown("Flujo: imagen â†’ preprocesamiento â†’ Xception â†’ Transfer Learning â†’ fine-tuning â†’ clasificaciÃ³n")
    st.caption("CIFAR-10 Â· 10 categorÃ­as Â· inferencia con model_final.keras")

    st.table({
        "ParÃ¡metro": [
            "Modelo", "Arquitectura base", "Pesos iniciales", "Entrada",
            "Salida", "Clasificador", "Top-k mostrado"
        ],
        "Valor": [
            "model_final.keras", "Xception", "ImageNet", "RGB 160Ã—160Ã—3",
            "10 clases", "Dense(10) + Softmax", "3"
        ],
    })
    st.info(
        "Esta interfaz utiliza el modelo final persistido para inferencia. "
        "No se ejecuta entrenamiento desde la aplicaciÃ³n."
    )

def render_training_evaluation() -> None:
    st.subheader("Entrenamiento y evaluaciÃ³n")
    st.markdown("### Estado del evaluador final")
    st.write(
        "El evaluador utilizado por la aplicaciÃ³n es **model_final.keras**, "
        "basado en Xception + Transfer Learning. En esta etapa no se realiza "
        "una nueva optimizaciÃ³n de precisiÃ³n."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Test CIFAR-10", "90.36%")
    c2.metric("Clases", "10")
    c3.metric("ImÃ¡genes de prueba", "10,000")
    st.markdown("### Prueba funcional de las 30 imÃ¡genes")
    st.write(
        "Las 30 imÃ¡genes precargadas se conservan como prueba funcional de "
        "la interfaz. No sustituyen la evaluaciÃ³n oficial."
    )
    st.warning(
        " como si pertenecieran "
        "al evaluador Xception."
    )

def render_about() -> None:
    st.subheader("Acerca de EikonorasÃ­s")
    st.write(
        "**EikonorasÃ­s CIFAR-10 akrivÃ­s** es una aplicaciÃ³n acadÃ©mica de "
        "visiÃ³n por computadora para clasificaciÃ³n multiclase de imÃ¡genes."
    )
    st.write(
        "El modelo evaluador utiliza Xception + Transfer Learning y "
        "preprocesamiento RGB a 160Ã—160."
    )
    st.write(
        "El dominio estÃ¡ limitado a las diez categorÃ­as oficiales de CIFAR-10."
    )
    st.write(
        "**Privacidad:** las imÃ¡genes cargadas se procesan para inferencia "
        "y no se requiere almacenamiento permanente de la imagen."
    )

def render_trivia() -> None:
    st.subheader("Trivia: Â¿de dÃ³nde viene EikonorasÃ­s?")
    st.markdown(
        """
**EikonorasÃ­s** es un neologismo creado para esta aplicaciÃ³n e inspirado
en dos raÃ­ces griegas:

- **EikÃ³n (Îµá¼°ÎºÏŽÎ½):** imagen, figura o representaciÃ³n.
- **Ã“rasis (á½…ÏÎ±ÏƒÎ¹Ï‚):** visiÃ³n o acto de ver.

El nombre puede entenderse libremente como **â€œvisiÃ³n de imÃ¡genesâ€**,
**â€œvisiÃ³n icÃ³nicaâ€** o **â€œiconovisiÃ³nâ€**.

### Â¿QuÃ© significa *akrivÃ­s*?

**AkrivÃ­s** es un tÃ©rmino de origen griego relacionado con **precisiÃ³n y exactitud**. En **EikonorasÃ­s CIFAR-10 akrivÃ­s**, identifica la versiÃ³n final del proyecto y expresa su enfoque en una clasificaciÃ³n **precisa, medible y evaluable**.

**Idea central:** *EikonorasÃ­s* representa la visiÃ³n; *CIFAR-10*, las diez categorÃ­as; y *akrivÃ­s*, el principio de precisiÃ³n que distingue la versiÃ³n final.

### Â¿QuÃ© representa el logo?

- **El ojo:** percepciÃ³n y visiÃ³n.
- **Los pÃ­xeles:** representaciÃ³n digital de la imagen.
- **Los nodos conectados:** red neuronal y aprendizaje profundo.
- **El dorado y grafito:** identidad visual de EikonorasÃ­s.
- **CIFAR-10:** dataset compuesto por diez categorÃ­as.
"""
    )

def render_help() -> None:
    st.subheader("Ayuda")
    help_tab1, help_tab2, help_tab3 = st.tabs(
        ["CÃ³mo funciona", "TecnologÃ­as", "Glosario"]
    )
    with help_tab1:
        st.markdown(
            """
### Inicio rÃ¡pido

1. Abre **Clasificar**.
2. Elige **ImÃ¡genes precargadas** o **Subir imagen**.
3. Selecciona o carga una imagen.
4. Revisa la previsualizaciÃ³n y confirma cuando corresponda.
5. Ejecuta el anÃ¡lisis.
6. Interpreta **clase estimada**, **confianza** y **Top-3**.

### ImÃ¡genes precargadas

Selecciona `img_test1` a `img_test30`, revisa la miniatura y la clase
esperada, confirma la imagen y ejecuta el anÃ¡lisis.

- **Clase esperada:** categorÃ­a conocida de la imagen de prueba.
- **Clase estimada:** categorÃ­a predicha por el evaluador.
- **Confianza:** probabilidad de la predicciÃ³n principal.
- **Top-3:** tres clases con mayor probabilidad.
- **EvaluaciÃ³n rÃ¡pida:** comparaciÃ³n esperado vs. predicho durante la sesiÃ³n.

Esta prueba es funcional y no sustituye la evaluaciÃ³n oficial de CIFAR-10.

### Imagen propia

Carga un JPG, JPEG o PNG vÃ¡lido. La aplicaciÃ³n valida el archivo, lo
convierte a RGB y lo prepara a **160Ã—160** antes de la inferencia.

### Confianza

**Confianza no significa certeza.** Softmax distribuye la probabilidad entre
las diez clases. Una imagen externa puede estar fuera de la distribuciÃ³n.

La app reconoce: **aviÃ³n, automÃ³vil, ave, gato, ciervo, perro, rana,
caballo, barco y camiÃ³n**.
"""
        )
    with help_tab2:
        st.markdown(
            """
### TecnologÃ­as

| Capa | TecnologÃ­a | FunciÃ³n |
|---|---|---|
| Interfaz | Streamlit | NavegaciÃ³n y resultados |
| ValidaciÃ³n | Python + Pillow | ValidaciÃ³n de imÃ¡genes |
| Preprocesamiento | Pillow + NumPy | RGB y 160Ã—160 |
| Modelo | TensorFlow / Keras | Inferencia |
| Arquitectura | Xception | ExtracciÃ³n de caracterÃ­sticas |
| Transfer Learning | ImageNet | Representaciones iniciales |
| Fine-tuning | Keras | AdaptaciÃ³n controlada |
| ClasificaciÃ³n | Dense(10) + Softmax | Diez probabilidades |

**Flujo:** Usuario â†’ Streamlit â†’ validaciÃ³n â†’ RGB 160Ã—160 â†’ Xception â†’
Transfer Learning â†’ fine-tuning â†’ Softmax â†’ predicciÃ³n.
"""
        )
    with help_tab3:
        st.markdown(
            """
### Glosario

- **CIFAR-10:** conjunto de diez categorÃ­as de imÃ¡genes.
- **CNN:** red neuronal convolucional para patrones visuales.
- **Xception:** arquitectura convolucional usada como modelo base.
- **Transfer Learning:** reutilizaciÃ³n de representaciones aprendidas.
- **Fine-tuning:** ajuste controlado del modelo para el dominio objetivo.
- **Inferencia:** ejecuciÃ³n del modelo sobre una imagen nueva.
- **Softmax:** convierte las salidas en probabilidades normalizadas.
- **Confianza:** probabilidad asociada a la clase predicha.
- **Top-3:** tres clases con mayor probabilidad.
- **Preprocesamiento:** transformaciÃ³n previa a la inferencia.
"""
        )

def render_institutional() -> None:
    """IdentificaciÃ³n acadÃ©mica e institucional del proyecto."""
    st.subheader('Institucional')
    st.caption('IdentificaciÃ³n acadÃ©mica de EikonorasÃ­s CIFAR-10 akrivÃ­s y del contexto formativo en el que fue desarrollado.')
    st.markdown('\n        <div class="analysis-box">\n        <strong style="font-size:.82rem;letter-spacing:.05em;">PROYECTO ACADÃ‰MICO</strong><br><br>\n        <strong style="font-size:1.35rem;">INSTITUTO INTERNACIONAL DE AGUASCALIENTES</strong><br>\n        MaestrÃ­a en Inteligencia Artificial para la TransformaciÃ³n Digital\n        </div>\n        ', unsafe_allow_html=True)
    st.link_button('Sitio oficial del Instituto Internacional de Aguascalientes', 'https://www.iinternacional.edu.mx/')
    st.markdown('### InformaciÃ³n acadÃ©mica')
    st.markdown('\n        **Asignatura:** Aprendizaje Profundo  \n        **AplicaciÃ³n:** EikonorasÃ­s CIFAR-10 akrivÃ­s  \n        **Proyecto:** DiseÃ±o, implementaciÃ³n, entrenamiento, evaluaciÃ³n y\n        despliegue web de una red neuronal convolucional en Python para la\n        clasificaciÃ³n multiclase de imÃ¡genes mediante CIFAR-10.\n        ')
    st.markdown('### AutorÃ­a acadÃ©mica')
    st.markdown('\n        **Alumno:** Antonio NicolÃ¡s Toro GonzÃ¡lez  \n        **Tutora:** Dra. Claudia Andrea Vidales Basurto\n        ')
    st.markdown('### DescripciÃ³n del proyecto')
    st.write('EikonorasÃ­s CIFAR-10 akrivÃ­s es una aplicaciÃ³n web interactiva desarrollada en Python que utiliza una Xception + Transfer Learning para clasificar imÃ¡genes en las diez categorÃ­as de CIFAR-10. El proyecto integra preparaciÃ³n y particiÃ³n de datos, preprocesamiento, data augmentation, diseÃ±o de la arquitectura, entrenamiento, validaciÃ³n, evaluaciÃ³n sobre un conjunto de prueba independiente, anÃ¡lisis de errores, inferencia y despliegue web.')
    st.markdown('### IdentificaciÃ³n tÃ©cnica')
    institutional_tech = pd.DataFrame({'Componente': ['Lenguaje', 'Modelo', 'Deep Learning', 'Preprocesamiento', 'EvaluaciÃ³n', 'Interfaz web', 'Dataset', 'Repositorio', 'AplicaciÃ³n web'], 'TecnologÃ­a / referencia': ['Python 3.11', 'Xception + Transfer Learning para clasificaciÃ³n multiclase', 'TensorFlow / Keras + Keras Applications', 'Pillow + NumPy', 'scikit-learn + pandas', 'Streamlit', 'CIFAR-10 â€” UCI Machine Learning Repository', 'https://github.com/edtech-mx-ve/eikonorasis-cifar10-akrivis', 'https://eikonorasis-cifar10.streamlit.app/']})
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
            '<div class="status status-warning">Modo demostraciÃ³n: se estÃ¡ usando el modelo smoke.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("No se encontrÃ³ un modelo entrenado.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Clasificar", "Dataset", "Modelo", "Entrenamiento y evaluaciÃ³n",
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
        '<div class="footer">EikonorasÃ­s CIFAR-10 akrivÃ­s Â· Proyecto acadÃ©mico de aprendizaje profundo</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


