# Eikonorasís CIFAR-10 akrivís

### Clasificación multiclase mediante Xception + Transfer Learning + fine-tuning controlado

Aplicación web académica desarrollada en Python, TensorFlow/Keras y Streamlit para clasificar imágenes en las diez categorías de CIFAR-10, mostrando clase estimada, confianza y Top-3.

- **Aplicación:** https://eikonorasis-cifar10-akrivis.streamlit.app/
- **Repositorio:** https://github.com/edtech-mx-ve/eikonorasis-cifar10-akrivis
- **Institución:** Instituto Internacional de Aguascalientes
- **Programa:** Maestría en Inteligencia Artificial para la Transformación Digital
- **Asignatura:** Aprendizaje Profundo

---

## 1. Estado actual

Esta es la versión actual **akrivís** del proyecto. Este README sustituye la documentación heredada de la aplicación anterior basada en una CNN propia.

| Elemento | Estado actual |
|---|---|
| Arquitectura | **Xception + Transfer Learning + fine-tuning controlado** |
| Modelo de inferencia | `model/model_final.keras` |
| Metadatos | `model/model_final.json` |
| Entrada de inferencia | RGB `160×160×3` |
| Salida | 10 clases CIFAR-10 |
| Validación | **91.10 %** |
| Test CIFAR-10 | **90.36 %** |
| Objetivo de referencia | **92 %** |
| Objetivo alcanzado | **No** |
| Streamlit | **1.65.0** |
| Python | **3.11** |

El resultado formal de test es **90.36 %**. El objetivo de 92 % es una referencia de evaluación y no debe presentarse como alcanzado.

---

## 2. ¿Qué hace la aplicación?

Eikonorasís clasifica imágenes dentro de las diez categorías oficiales de CIFAR-10:

**Avión · Automóvil · Ave · Gato · Ciervo · Perro · Rana · Caballo · Barco · Camión**

La interfaz permite:

- seleccionar imágenes precargadas;
- subir imágenes propias;
- validar las entradas;
- convertir las imágenes a RGB;
- redimensionarlas a `160×160`;
- ejecutar el modelo Xception;
- mostrar la clase estimada;
- mostrar la confianza;
- mostrar el Top-3;
- consultar información del dataset;
- consultar información del modelo;
- consultar evaluación y limitaciones;
- consultar ayuda y documentación del proyecto.

La aplicación es un demostrador académico. No es un clasificador universal de imágenes.

> Una probabilidad Softmax alta expresa la preferencia del modelo entre las diez clases conocidas; no garantiza que una imagen externa pertenezca realmente al dominio de CIFAR-10.

---

## 3. Flujo de inferencia

```text
Imagen
  ↓
Validación
  ↓
RGB
  ↓
Resize 160×160
  ↓
Xception + Transfer Learning
  ↓
Fine-tuning / clasificador
  ↓
Softmax: 10 probabilidades
  ↓
Clase + confianza + Top-3
```

El entrenamiento y la inferencia son procesos separados. La aplicación utiliza el modelo persistido y no necesita volver a entrenarlo para cada predicción.

---

## 4. Modelo actual

El evaluador final utiliza **Xception**, inicialmente preentrenada con pesos de **ImageNet**, adaptada mediante Transfer Learning y fine-tuning controlado.

### Configuración

| Elemento | Configuración |
|---|---|
| Arquitectura base | Xception |
| Pesos iniciales | ImageNet |
| Entrada | RGB `160×160×3` |
| Salida | 10 categorías |
| Modelo | `model/model_final.keras` |
| Metadatos | `model/model_final.json` |

El archivo `model/model_final.json` conserva metadatos del modelo, incluyendo arquitectura, época seleccionada, resultados de validación y test, tamaño de entrada, preprocesamiento y SHA-256 del artefacto.

---

## 5. Evaluación formal

Resultados documentados para el modelo actual:

| Métrica | Resultado |
|---|---:|
| Accuracy de validación | **91.10 %** |
| Accuracy de test CIFAR-10 | **90.36 %** |
| Objetivo | **92.00 %** |
| Objetivo alcanzado | **No** |

El **90.36 %** es el resultado formal que debe utilizarse para describir el desempeño del modelo.

No debe confundirse con las pruebas funcionales realizadas desde la interfaz.

---

## 6. Pruebas funcionales

La aplicación incluye imágenes precargadas en:

```text
assets/test_images/
```

Su propósito es verificar el flujo:

```text
carga → validación → preprocesamiento → inferencia → resultado
```

Las pruebas funcionales son independientes de la evaluación formal sobre el conjunto de test de CIFAR-10.

Por tanto:

- **evaluación formal** = desempeño del modelo;
- **prueba funcional** = funcionamiento de la aplicación.

Una prueba funcional no sustituye la métrica formal del conjunto de prueba.

---

## 7. CIFAR-10

CIFAR-10 contiene:

- 60,000 imágenes RGB;
- resolución original de `32×32` píxeles;
- 10 categorías;
- 50,000 imágenes de entrenamiento;
- 10,000 imágenes de prueba.

La resolución original del dataset **no debe confundirse** con la entrada del modelo actual. Para Xception, la imagen se adapta a:

```text
160×160×3
```

### Fuente

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/691/cifar+10

DOI:

https://doi.org/10.24432/C5889J

---

## 8. Estructura actual del repositorio

```text
eikonorasis-cifar10-akrivis/
│
├── app.py
├── README.md
├── requirements.txt
│
├── .streamlit/
│   └── config.toml
│
├── .devcontainer/
│   └── devcontainer.json
│
├── assets/
│   ├── logo.png
│   ├── flujo_clasificacion_xception.png
│   ├── flujo_evaluador_xception.png
│   ├── modelo_xception_eikonorasis.svg
│   ├── modelo_xception_transfer_learning.svg
│   └── test_images/
│       ├── manifest.csv
│       └── img_test*.png
│
├── model/
│   ├── model_final.keras
│   └── model_final.json
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── inference.py
    ├── logging_config.py
    ├── pilot_data.py
    ├── preprocessing.py
    ├── transfer_config.py
    ├── transfer_data.py
    ├── transfer_model.py
    ├── transfer_reports.py
    └── transfer_training.py
```

### Responsabilidades principales

| Componente | Responsabilidad |
|---|---|
| `app.py` | Interfaz Streamlit y flujo de interacción |
| `src/config.py` | Configuración y constantes |
| `src/inference.py` | Carga y ejecución del modelo |
| `src/preprocessing.py` | Validación y preparación de imágenes |
| `src/logging_config.py` | Logging |
| `src/data.py` | Gestión de datos |
| `src/pilot_data.py` | Datos de pruebas piloto |
| `src/transfer_config.py` | Configuración de Transfer Learning |
| `src/transfer_data.py` | Preparación de datos |
| `src/transfer_model.py` | Construcción/configuración del modelo |
| `src/transfer_training.py` | Entrenamiento |
| `src/transfer_reports.py` | Reportes |
| `model/model_final.keras` | Modelo para inferencia |
| `model/model_final.json` | Metadatos del modelo |
| `assets/test_images/` | Imágenes precargadas |

---

## 9. Instalación local

Se recomienda Python 3.11.

### Crear entorno

```powershell
py -3.11 -m venv .venv
```

### Activar

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Verificar

```powershell
python --version
```

### Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Ejecución local

Verificar sintaxis:

```powershell
python -m py_compile .pp.py
```

Ejecutar:

```powershell
python -m streamlit run .pp.py
```

Abrir:

```text
http://localhost:8501
```

La aplicación debe cargar `model/model_final.keras` y permitir inferencia con imágenes precargadas o proporcionadas por el usuario.

---

## 11. Verificaciones del código

Para `app.py`:

```powershell
python -m py_compile .pp.py
```

Para los módulos del pipeline:

```powershell
python -m py_compile `
    .\src\data.py `
    .\src\pilot_data.py `
    .\src	ransfer_config.py `
    .\src	ransfer_data.py `
    .\src	ransfer_model.py `
    .\src	ransfer_reports.py `
    .\src	ransfer_training.py
```

Para Git:

```powershell
git diff --check
git status --short
```

Estas comprobaciones verifican sintaxis e higiene básica del repositorio; no sustituyen la evaluación del modelo.

---

## 12. Dependencias y despliegue

Las versiones de Python y librerías deben tomarse de `requirements.txt`.

La versión de Streamlit fijada para el despliegue actual es:

```text
Streamlit 1.65.0
```

Repositorio:

https://github.com/edtech-mx-ve/eikonorasis-cifar10-akrivis

Rama:

```text
main
```

Archivo de entrada:

```text
app.py
```

Aplicación:

https://eikonorasis-cifar10-akrivis.streamlit.app/

---

## 13. Control de versiones

Flujo recomendado:

```powershell
git status
git diff
python -m py_compile .pp.py
git diff --check
git add <archivos>
git diff --cached --check
git commit -m "Descripción del cambio"
git push origin main
```

Antes de publicar, revisar que no se incorporen archivos locales, secretos, credenciales o entornos virtuales.

No publicar:

```text
.venv/
__pycache__/
*.pyc
.env
secretos
credenciales
archivos temporales
```

---

## 14. Robustez y seguridad

La aplicación incorpora:

- validación de entradas;
- validación del contenido de las imágenes;
- conversión controlada a RGB;
- manejo de errores;
- logging;
- separación entre interfaz, configuración, preprocesamiento e inferencia;
- modelo persistido;
- exclusión de secretos y archivos locales mediante Git.

Las imágenes suministradas por usuarios deben considerarse entradas no confiables y validarse antes de procesarse.

La aplicación es académica y demostrativa y no debe utilizarse para decisiones de alto impacto.

---

## 15. Reproducibilidad

El proyecto conserva componentes del pipeline de Transfer Learning para mantener trazabilidad y permitir continuar el desarrollo.

El archivo:

```text
model/model_final.json
```

documenta información del modelo final, incluyendo:

- arquitectura;
- época seleccionada;
- accuracy de validación;
- accuracy de test;
- objetivo de accuracy;
- tamaño de entrada;
- forma de salida;
- preprocesamiento;
- SHA-256.

El modelo utilizado por la aplicación es:

```text
model/model_final.keras
```

---

## 16. Limitaciones

- Solo se reconocen las diez clases de CIFAR-10.
- CIFAR-10 contiene imágenes originales de `32×32`.
- El modelo actual recibe entradas adaptadas a `160×160×3`.
- Imágenes externas pueden diferir de la distribución de entrenamiento.
- Una confianza Softmax alta no garantiza corrección.
- El accuracy formal de test es **90.36 %**, inferior al objetivo de referencia de **92 %**.
- Las imágenes precargadas sirven para pruebas funcionales y no sustituyen el test oficial.
- El sistema es un proyecto académico y demostrativo.

---

## 17. Contexto académico

**Instituto Internacional de Aguascalientes**

**Programa:** Maestría en Inteligencia Artificial para la Transformación Digital

**Asignatura:** Aprendizaje Profundo

**Proyecto:** Eikonorasís CIFAR-10 akrivís

El proyecto integra aprendizaje profundo, visión por computadora, Transfer Learning, fine-tuning, clasificación multiclase, evaluación, ingeniería de software y despliegue de una aplicación de IA.

---

## 18. Fuentes

- UCI Machine Learning Repository — CIFAR-10: https://archive.ics.uci.edu/dataset/691/cifar+10
- DOI CIFAR-10: https://doi.org/10.24432/C5889J
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Streamlit: https://streamlit.io/
- scikit-learn: https://scikit-learn.org/

---

## 19. Enlaces del proyecto

- **Aplicación:** https://eikonorasis-cifar10-akrivis.streamlit.app/
- **Repositorio:** https://github.com/edtech-mx-ve/eikonorasis-cifar10-akrivis
- **CIFAR-10:** https://archive.ics.uci.edu/dataset/691/cifar+10
- **Instituto Internacional de Aguascalientes:** https://www.iinternacional.edu.mx/

---

<div align="center">

### Eikonorasís CIFAR-10 akrivís

**Imagen → Xception + Transfer Learning → Softmax → Clase + Confianza + Top-3**

**Accuracy formal de test: 90.36 %**

Proyecto académico de Deep Learning y visión por computadora.

</div>
