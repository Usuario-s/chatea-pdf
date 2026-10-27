import platform
import hashlib

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Chat PDF - RAG",
    page_icon="💬",
    layout="wide"
)


# ==========================================
# DISEÑO VISUAL BLANCO Y AZUL
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F8FF;
    color: #193153;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.chat-header {
    background: linear-gradient(135deg, #1264D8, #398BFA);
    padding: 30px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(28, 100, 216, 0.15);
}

.chat-header h1 {
    color: white;
    font-size: 30px;
    font-weight: 750;
    margin-bottom: 8px;
}

.chat-header p {
    color: #EAF3FF;
    font-size: 15px;
    margin-bottom: 0;
}

section[data-testid="stSidebar"] {
    background: #EAF2FF;
    border-right: 1px solid #D5E5FC;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: #193D70;
}

h2, h3, h4 {
    color: #174A91;
}

p, label {
    color: #294565;
}

.stTextInput input,
.stTextArea textarea {
    background-color: white !important;
    color: #193153 !important;
    border: 1px solid #C8D9F1 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #2678E8 !important;
    box-shadow: 0 0 0 2px rgba(38, 120, 232, 0.12) !important;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(135deg, #1769D2, #398BFA);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 650;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #1055B5;
    border: none;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #8BB8F2;
    border-radius: 15px;
    padding: 14px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border: 1px solid #DCE8F8;
    border-radius: 18px;
    padding: 12px;
}

hr {
    border-color: #DCE8F8;
}

.user-message {
    background: #E1EEFF;
    border: 1px solid #C5DDFF;
    border-radius: 18px 5px 18px 18px;
    padding: 16px 20px;
    margin: 15px 0 20px 10%;
    color: #174A91;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

.chat-label {
    color: #2678E8;
    font-size: 13px;
    font-weight: 750;
    margin-bottom: 8px;
}

.chat-footer {
    text-align: center;
    color: #7B91AD;
    font-size: 12px;
    padding: 18px 0 5px 0;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .chat-header {
        padding: 22px;
    }

    .chat-header h1 {
        font-size: 24px;
    }

    .user-message {
        margin-left: 3%;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# ENCABEZADO
# ==========================================

st.markdown("""
<div class="chat-header">
    <h1>💬 Chat PDF</h1>
    <p>
        Generación Aumentada por Recuperación (RAG)<br>
        Conversa con tus documentos, consulta información
        y encuentra respuestas basadas en el contenido del PDF.
    </p>
</div>
""", unsafe_allow_html=True)

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
    st.subheader("📚 Tu asistente documental")

    st.write(
        "Este agente te ayudará a realizar análisis "
        "sobre el PDF cargado."
    )

    st.divider()

    st.markdown("### ✨ ¿Qué puedes hacer?")

    st.markdown("""
    - 📄 Cargar un documento PDF.
    - 🔎 Buscar información específica.
    - 💡 Identificar ideas principales.
    - 📖 Consultar fragmentos y páginas.
    """)

    st.divider()
    st.caption("Chat PDF · Sistema RAG")


# ==========================================
# CLAVE API
# ==========================================

st.markdown("### 🔑 Conexión con OpenAI")

api_key = st.text_input(
    "Ingresa tu clave de API de OpenAI",
    type="password",
    placeholder="sk-...",
    help="Tu clave se utiliza para generar embeddings y respuestas."
)


# ==========================================
# CARGAR PDF
# ==========================================

st.markdown("### 📄 Documento de consulta")

pdf = st.file_uploader(
    "Carga el archivo PDF",
    type=["pdf"],
    help="Selecciona el documento que quieres analizar."
)


# ==========================================
# CREAR BASE VECTORIAL
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

    return FAISS.from_documents(
        fragmentos,
        embeddings
    )


# ==========================================
# VALIDAR ENTRADAS
# ==========================================

if not api_key:
    st.info("🔐 Ingresa tu clave de API de OpenAI para continuar.")

elif pdf is None:
    st.info("📂 Carga un archivo PDF para comenzar.")

else:
    try:
        # ==========================================
        # EXTRAER TEXTO DEL PDF
        # ==========================================

        pdf.seek(0)
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

        st.success(
            f"📄 Texto extraído: {len(texto_completo):,} caracteres"
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
            f"🧩 Documento dividido en {len(fragmentos)} fragmentos."
        )

        # ==========================================
        # CREAR BASE DE CONOCIMIENTO
        # ==========================================

        identificador = hashlib.sha256(
            texto_completo.encode("utf-8")
        ).hexdigest()

        with st.spinner("🔄 Procesando el documento..."):
            base_conocimiento = crear_base_vectorial(
                fragmentos,
                api_key,
                identificador
            )

        st.success("✅ Base de conocimiento lista.")
        st.divider()

        # ==========================================
        # FORMULARIO DE PREGUNTAS
        # ==========================================

        st.markdown("## 💬 Conversación con tu documento")
        st.caption(f"Documento activo: {pdf.name}")

        with st.container(border=True):
            st.markdown("### 👋 ¿Qué quieres saber?")

            with st.form("formulario_pregunta"):
                pregunta = st.text_area(
                    "Escribe tu pregunta",
                    placeholder=(
                        "Ejemplo: ¿Cuáles son las ideas principales "
                        "del documento?"
                    ),
                    height=110
                )

                consultar = st.form_submit_button(
                    "🔎 Consultar documento",
                    use_container_width=True
                )

        # ==========================================
        # GENERAR RESPUESTA
        # ==========================================

        if consultar:
            if not pregunta.strip():
                st.warning("Escribe una pregunta antes de continuar.")

            else:
                try:
                    st.markdown(
                        f"""
                        <div class="user-message">
                            <div class="chat-label">TÚ</div>
                            {pregunta}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.spinner("🤖 Analizando el documento..."):
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

                    # ==========================================
                    # MOSTRAR RESPUESTA
                    # ==========================================

                    st.markdown(
                        '<div class="chat-label">🤖 ASISTENTE PDF</div>',
                        unsafe_allow_html=True
                    )

                    with st.container(border=True):
                        st.markdown(respuesta.content)

                    # ==========================================
                    # MOSTRAR FUENTES
                    # ==========================================

                    with st.expander("📚 Ver fragmentos consultados"):
                        for indice, documento in enumerate(
                            documentos_relevantes,
                            start=1
                        ):
                            st.markdown(
                                f"**📄 Fragmento {indice} | "
                                f"Página {documento.metadata.get('pagina', 'N/D')}**"
                            )

                            st.write(documento.page_content)
                            st.divider()

                except Exception as error_pregunta:
                    st.error(
                        f"Error al responder la pregunta: {error_pregunta}"
                    )

        # ==========================================
        # PIE DE PÁGINA
        # ==========================================

        st.markdown("""
        <div class="chat-footer">
            Chat PDF · RAG · Respuestas basadas en el documento cargado
        </div>
        """, unsafe_allow_html=True)

    except Exception as error_pdf:
        st.error(f"Error al procesar el PDF: {error_pdf}")
