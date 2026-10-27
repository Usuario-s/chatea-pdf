import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --------------------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="Chat PDF - RAG",
    page_icon="💬",
    layout="wide"
)

st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.write("Versión de Python:", platform.python_version())

# --------------------------------------------------
# CARGAR IMAGEN
# --------------------------------------------------

try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)
except Exception:
    st.warning(
        "No se pudo cargar Chat_pdf.png. "
        "Verifica que la imagen esté en el repositorio."
    )

# --------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------

with st.sidebar:
    st.subheader(
        "Este agente te ayudará a realizar análisis "
        "sobre el PDF cargado."
    )

    st.write(
        "Carga un documento y escribe preguntas "
        "para consultar su contenido."
    )

# --------------------------------------------------
# CLAVE DE API
# --------------------------------------------------

ke = st.text_input(
    "Ingresa tu clave de API de OpenAI",
    type="password"
)

# --------------------------------------------------
# CARGAR DOCUMENTO PDF
# --------------------------------------------------

pdf = st.file_uploader(
    "Carga el archivo PDF",
    type=["pdf"]
)

# --------------------------------------------------
# PROCESAR DOCUMENTO
# --------------------------------------------------

if not ke:
    st.warning(
        "Por favor, ingresa tu clave de API de OpenAI "
        "para continuar."
    )

elif pdf is None:
    st.info("Por favor, carga un archivo PDF para comenzar.")

else:
    try:
        # Extraer texto de las páginas
        pdf_reader = PdfReader(pdf)

        documentos = []

        for numero_pagina, pagina in enumerate(
            pdf_reader.pages,
            start=1
        ):
            texto_pagina = pagina.extract_text() or ""

            if texto_pagina.strip():
                documentos.append(
                    Document(
                        page_content=texto_pagina,
                        metadata={
                            "pagina": numero_pagina,
                            "archivo": pdf.name
                        }
                    )
                )

        # Comprobar que el PDF tenga texto extraíble
        if not documentos:
            st.error(
                "No se pudo extraer texto del PDF. "
                "Si es un documento escaneado, "
                "necesitas aplicar OCR primero."
            )
            st.stop()

        texto_completo = "\n".join(
            documento.page_content
            for documento in documentos
        )

        st.info(
            f"Archivo: {pdf.name} | "
            f"Texto extraído: {len(texto_completo):,} caracteres"
        )

        # --------------------------------------------------
        # DIVIDIR EL TEXTO EN FRAGMENTOS
        # --------------------------------------------------

        separador = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        fragmentos = separador.split_documents(documentos)

        if not fragmentos:
            st.error(
                "No fue posible dividir el documento en fragmentos."
            )
            st.stop()

        st.success(
            f"Documento dividido en {len(fragmentos)} fragmentos."
        )

        # --------------------------------------------------
        # CREAR LA BASE DE CONOCIMIENTO VECTORIAL
        # --------------------------------------------------

        @st.cache_resource(show_spinner=False)
        def crear_base_vectorial(
            fragmentos,
            clave_api,
            nombre_archivo,
            texto_documento
        ):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=clave_api
            )

            base = FAISS.from_documents(
                fragmentos,
                embeddings
            )

            return base

        with st.spinner(
            "Creando la base de conocimiento del documento..."
        ):
            base_conocimiento = crear_base_vectorial(
                fragmentos,
                ke,
                pdf.name,
                texto_completo
            )

        st.success("Base de conocimiento lista para consultar.")

        # --------------------------------------------------
        # INTERFAZ DE PREGUNTAS
        # --------------------------------------------------

        st.subheader(
            "Escribe qué quieres saber sobre el documento"
        )

        with st.form("formulario_pregunta"):
            pregunta = st.text_area(
                "Tu pregunta",
                placeholder=(
                    "Ejemplo
