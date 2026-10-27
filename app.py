import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ==========================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================

st.set_page_config(
    page_title="Chat PDF - RAG",
    page_icon="💬",
    layout="wide"
)

st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.write("Versión de Python:", platform.python_version())


# ==========================================
# CARGAR IMAGEN
# ==========================================

try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)
except Exception:
    st.warning("No se pudo cargar la imagen Chat_pdf.png.")


# ==========================================
# BARRA LATERAL
# ==========================================

with st.sidebar:
    st.subheader(
        "Este agente te ayudará a realizar análisis sobre el PDF cargado."
    )


# ==========================================
# CLAVE API DE OPENAI
# ==========================================

api_key = st.text_input(
    "Ingresa tu clave de API de OpenAI",
    type="password"
)


# ==========================================
# CARGAR ARCHIVO PDF
# ==========================================

pdf = st.file_uploader(
    "Carga el archivo PDF",
    type=["pdf"]
)


# ==========================================
# VALIDAR ENTRADAS
# ==========================================

if not api_key:
    st.warning("Ingresa tu clave de API de OpenAI para continuar.")

elif pdf is None:
    st.info("Carga un archivo PDF para comenzar.")

else:
    try:
        # ==========================================
        # EXTRAER TEXTO DEL PDF
        # ==========================================

        lector_pdf = PdfReader(pdf)
        documentos = []

        for numero_pagina, pagina in enumerate(
            lector_pdf.pages,
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

        if not documentos:
            st.error(
                "No se pudo extraer texto del PDF. "
                "Si es un documento escaneado, necesitas utilizar OCR."
            )
            st.stop()

        texto_completo = "\n".join(
            documento.page_content
            for documento in documentos
        )

        st.info(
            f"Texto extraído: {len(texto_completo)} caracteres"
        )

        # ==========================================
        # DIVIDIR TEXTO EN FRAGMENTOS
        # ==========================================

        divisor_texto = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        fragmentos = divisor_texto.split_documents(documentos)

        if not fragmentos:
            st.error("No se pudieron crear fragmentos del documento.")
            st.stop()

        st.success(
            f"Documento dividido en {len(fragmentos)} fragmentos."
        )

        # ==========================================
        # CREAR EMBEDDINGS Y BASE VECTORIAL
        # ==========================================

        @st.cache_resource(show_spinner=False)
        def crear_base_vectorial(
            fragmentos,
            api_key,
            identificador_documento
        ):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            base_vectorial = FAISS.from_documents(
                fragmentos,
                embeddings
            )

            return base_vectorial

        identificador = (
            f"{pdf.name}_{len(texto_completo)}_"
            f"{hash(texto_completo)}"
        )

        with st.spinner("Procesando el documento..."):
            base_conocimiento = crear_base_vectorial(
                fragmentos,
                api_key,
                identificador
            )

        st.success("Base de conocimiento lista.")

        # ==========================================
        # FORMULARIO DE PREGUNTAS
        # ==========================================

        st.subheader("¿Qué quieres saber sobre el documento?")

        with st.form("formulario_pregunta"):
            pregunta = st.text_area(
                "Escribe tu pregunta",
                placeholder="Ejemplo: ¿Cuáles son las ideas principales?"
            )

            consultar = st.form_submit_button("Consultar documento")

        # ==========================================
        # GENERAR RESPUESTA
        # ==========================================

        if consultar:
            if not pregunta.strip():
                st.warning("Escribe una pregunta antes de continuar.")

            else:
                try:
                    with st.spinner("Buscando información..."):

                        documentos_relevantes = (
                            base_conocimiento.similarity_search(
                                pregunta,
                                k=4
                            )
                        )

                        contexto = "\n\n".join(
                            (
                                f"Página {documento.metadata.get('pagina', 'N/D')}:\n"
                                f"{documento.page_content}"
                            )
                            for documento in documentos_relevantes
                        )

                        modelo = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0,
                            api_key=api_key
                        )

                        mensajes = [
                            (
                                "system",
                                "Eres un asistente especializado en "
                                "analizar documentos PDF. Responde en "
                                "español utilizando el contexto proporcionado. "
                                "Si la información no aparece en el contexto, "
                                "indícalo claramente. No inventes información. "
                                "Cuando sea posible, menciona las páginas "
                                "que respaldan tu respuesta.\n\n"
                                f"CONTEXTO DEL DOCUMENTO:\n{contexto}"
                            ),
                            ("human", pregunta)
                        ]

                        respuesta = modelo.invoke(mensajes)

                    # Mostrar respuesta
                    st.markdown("### Respuesta")
                    st.markdown(respuesta.content)

                    # ==========================================
                    # MOSTRAR FUENTES
                    # ==========================================

                    with st.expander("Ver fragmentos consultados"):
                        for indice, documento in enumerate(
                            documentos_relevantes,
                            start=1
                        ):
                            st.markdown(
                                f"**Fragmento {indice} | "
                                f"Página {documento.metadata.get('pagina', 'N/D')}**"
                            )

                            st.write(documento.page_content)
                            st.divider()

                except Exception as error_pregunta:
                    st.error(
                        f"Error al responder la pregunta: {error_pregunta}"
                    )

    except Exception as error_pdf:
        st.error(f"Error al procesar el PDF: {error_pdf}")
