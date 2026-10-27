import platform
import hashlib
import html

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
# DISEÑO: CHAT BLANCO Y AZUL
# ==========================================

st.markdown("""
<style>
/* Fondo blanco y texto negro */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #FFFFFF !important;
    color: #111111 !important;
}

html, body, [class*="css"] {
    color: #111111;
}

[data-testid="stHeader"] {
    background: #FFFFFF;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Área principal */
.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Todos los textos */
h1, h2, h3, h4, h5, h6,
p, label, span, li,
.stMarkdown, .stCaption,
[data-testid="stMarkdownContainer"] {
    color: #111111;
}

/* Encabezado del chat */
.chat-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 23px 26px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.035);
}

.chat-logo {
    min-width: 52px;
    height: 52px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #EAF2FF;
    border-radius: 16px;
    font-size: 27px;
}

.chat-header h1 {
    color: #111111 !important;
    font-size: 28px;
    font-weight: 750;
    margin: 0;
}

.chat-header p {
    color: #555555 !important;
    font-size: 14px;
    margin: 5px 0 0 0;
}

/* Barra lateral */
section[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E5E7EB;
}

section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

.sidebar-card {
    background: #FFFFFF;
    border: 1px solid #E1E7EF;
    border-radius: 15px;
    padding: 16px;
    margin: 12px 0;
}

/* Separadores */
hr {
    border-color: #E5E7EB !important;
}

/* Campos de texto */
.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 13px !important;
    padding: 12px !important;
    caret-color: #2563EB;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #737373 !important;
    opacity: 1;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.13) !important;
}

/* Botones */
.stButton button,
.stFormSubmitButton button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 12px !important;
    font-weight: 650 !important;
    padding: 10px 18px !important;
    transition: all 0.2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18);
}

/* Carga de archivos */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #93B4F5;
    border-radius: 15px;
    padding: 13px;
}

[data-testid="stFileUploader"] * {
    color: #111111 !important;
}

/* Indicadores */
[data-testid="stAlert"] {
    border-radius: 13px;
}

/* Paneles */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 17px;
    padding: 13px;
}

/* Zona de conversación */
.conversation-title {
    font-size: 23px;
    font-weight: 750;
    color: #111111;
    margin: 8px 0 5px 0;
}

.conversation-subtitle {
    font-size: 13px;
    color: #666666;
    margin-bottom: 22px;
}

/* Burbuja del usuario */
.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 22px 0;
}

.user-bubble {
    max-width: 78%;
    background: #EAF2FF;
    color: #111111 !important;
    border: 1px solid #D8E5FF;
    border-radius: 20px 20px 5px 20px;
    padding: 15px 19px;
    line-height: 1.65;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

/* Mensaje del asistente */
.assistant-row {
    display: flex;
    align-items: flex-start;
    gap: 13px;
    margin: 20px 0;
}

.assistant-avatar {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    width: 39px;
    height: 39px;
    background: #2563EB;
    color: #FFFFFF !important;
    border-radius: 13px;
    font-size: 19px;
}

.assistant-bubble {
    flex: 1;
    min-width: 0;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 5px 18px 18px 18px;
    padding: 17px 20px;
    line-height: 1.75;
    color: #111111 !important;
    overflow-wrap: anywhere;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.025);
}

.assistant-bubble p,
.assistant-bubble li,
.assistant-bubble h1,
.assistant-bubble h2,
.assistant-bubble h3,
.assistant-bubble strong {
    color: #111111 !important;
}

.message-label {
    color: #2563EB !important;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.4px;
    margin-bottom: 8px;
}

/* Área de pregunta */
.question-panel {
    background: #FFFFFF;
    border: 1px solid #DCE5F1;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.04);
}

/* Fuentes */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 13px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #111111 !important;
}

/* Pie de página */
.chat-footer {
    text-align: center;
    color: #737373 !important;
    font-size: 12px;
    padding: 28px 0 8px 0;
}

/* Adaptación móvil */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .chat-header {
        padding: 17px;
    }

    .chat-header h1 {
        font-size: 23px;
    }

    .user-bubble {
        max-width: 92%;
    }

    .assistant-bubble {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# ENCABEZADO
# ==========================================

st.markdown("""
<div class="chat-header">
    <div class="chat-logo">💬</div>
    <div>
        <h1>Chat PDF</h1>
        <p>Tu asistente inteligente para analizar documentos PDF</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"Python {platform.python_version()}")


# ==========================================
# IMAGEN
# ==========================================

try:
    image = Image.open("im1.jpg")
    st.image(image, width=300)
except Exception:
    pass


# ==========================================
# BARRA LATERAL
# ==========================================

with st.sidebar:
    st.markdown("## 💬 Chat PDF")
    st.write("Consulta tus documentos con inteligencia artificial.")

    st.divider()

    st.markdown("### ✨ Funciones")

    st.markdown("""
    - 📄 Analizar documentos PDF.
    - 🔎 Buscar información.
    - 💡 Encontrar ideas principales.
    - 📚 Consultar fragmentos y páginas.
    """)

    st.divider()

    st.markdown("""
    <div class="sidebar-card">
        <strong>🔵 Sistema RAG</strong>
        <p>Las respuestas se generan utilizando fragmentos
        recuperados del documento que cargaste.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# CLAVE API
# ==========================================

st.markdown("### 🔑 Conexión con OpenAI")

api_key = st.text_input(
    "Clave de API de OpenAI",
    type="password",
    placeholder="Ingresa tu clave de API",
    help="Se utiliza para generar embeddings y respuestas."
)


# ==========================================
# CARGAR PDF
# ==========================================

st.markdown("### 📄 Tu documento")

pdf = st.file_uploader(
    "Selecciona un archivo PDF",
    type=["pdf"]
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
    st.info("Ingresa tu clave de API para comenzar.")

elif pdf is None:
    st.info("Carga un PDF para empezar a conversar con tu documento.")

else:
    try:
        # ==========================================
        # EXTRAER TEXTO
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
                "Si es un documento escaneado, necesitarás OCR."
            )
            st.stop()

        texto_completo = "\n".join(
            documento.page_content
            for documento in documentos
        )

        st.success(
            f"Documento leído: {len(texto_completo):,} caracteres."
        )

        # ==========================================
        # DIVIDIR TEXTO
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

        st.caption(f"Fragmentos creados: {len(fragmentos)}")

        # ==========================================
        # BASE DE CONOCIMIENTO
        # ==========================================

        identificador = hashlib.sha256(
            texto_completo.encode("utf-8")
        ).hexdigest()

        with st.spinner("Preparando tu documento..."):
            base_conocimiento = crear_base_vectorial(
                fragmentos,
                api_key,
                identificador
            )

        # ==========================================
        # ÁREA DE CHAT
        # ==========================================

        st.divider()

        st.markdown(
            '<div class="conversation-title">Conversación</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="conversation-subtitle">'
            f'Consultando: {html.escape(pdf.name)}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ==========================================
        # FORMULARIO
        # ==========================================

        with st.form("formulario_pregunta"):
            pregunta = st.text_area(
                "Escribe tu mensaje",
                placeholder=(
                    "Pregunta lo que quieras sobre el documento..."
                ),
                height=100
            )

            consultar = st.form_submit_button(
                "Enviar pregunta  ➤",
                use_container_width=True
            )

        # ==========================================
        # RESPONDER
        # ==========================================

        if consultar:
            if not pregunta.strip():
                st.warning("Escribe una pregunta antes de enviarla.")

            else:
                try:
                    # Mostrar el mensaje de forma segura.
                    pregunta_segura = html.escape(pregunta)

                    st.markdown(
                        f"""
                        <div class="user-message">
                            <div class="user-bubble">
                                {pregunta_segura}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.spinner("El asistente está pensando..."):
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

                    # Mostrar respuesta del asistente.
                    st.markdown(
                        """
                        <div class="assistant-row">
                            <div class="assistant-avatar">✦</div>
                            <div style="flex: 1; min-width: 0;">
                                <div class="message-label">
                                    ASISTENTE PDF
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.container(border=True):
                        st.markdown(respuesta.content)

                    # ==========================================
                    # FUENTES CONSULTADAS
                    # ==========================================

                    with st.expander("📚 Consultar fuentes del documento"):
                        for indice, documento in enumerate(
                            documentos_relevantes,
                            start=1
                        ):
                            pagina = documento.metadata.get(
                                "pagina",
                                "N/D"
                            )

                            st.markdown(
                                f"**Fragmento {indice} · Página {pagina}**"
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
            Chat PDF · Inteligencia artificial basada en tus documentos
        </div>
        """, unsafe_allow_html=True)

    except Exception as error_pdf:
        st.error(f"Error al procesar el PDF: {error_pdf}")
