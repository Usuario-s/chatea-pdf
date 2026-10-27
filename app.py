```python
import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuración de la aplicación
st.set_page_config(
    page_title="Chat PDF - RAG",
    page_icon="💬",
    layout="wide"
)

st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.write("Versión de Python:", platform.python_version())

# Cargar imagen
try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)
except Exception:
    st.warning("No se pudo cargar la imagen Chat_pdf.png.")

# Información lateral
with st.sidebar:
    st.subheader(
        "Este agente te ayudará a realizar análisis "
        "sobre el PDF cargado."
    )

# Clave de API
ke = st.text_input(
    "Ingresa tu clave de API de OpenAI",
    type="password"
)

# Cargar PDF
pdf = st.file_uploader(
    "Carga el archivo PDF",
    type=["pdf"]
)

if not ke:
    st.warning("Ingresa tu clave de API de OpenAI para continuar.")

elif pdf is None:
    st.info("Carga un archivo PDF para comenzar.")

else:
    try:
        # Leer el PDF
        pdf_reader = PdfReader(pdf)
        paginas = []

        for numero, page in enumerate(pdf_reader.pages, start=1):
            contenido = page.extract_text() or ""

            if contenido.strip():
                paginas.append(
                    Document(
                        page_content=contenido,
                        metadata={"pagina": numero}
                    )
                )

        if not paginas:
            st.error(
                "No se pudo extraer texto del PDF. "
                "Si es un documento escaneado, necesitarás OCR."
            )
            st.stop()

        texto = "\n".join(
            documento.page_content for documento in paginas
        )

        st.info(f"Texto extraído: {len(texto)} caracteres")

        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        chunks = text_splitter.split_documents(paginas)

        st.success(
            f"Documento dividido en {len(chunks)} fragmentos."
        )

        # Crear embeddings y base vectorial
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=ke
        )

        knowledge_base = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Preguntas del usuario
        st.subheader("¿Qué quieres saber sobre el documento?")

        user_question = st.text_area(
            "Escribe tu pregunta",
            placeholder="Ejemplo: ¿Cuáles son las ideas principales?",
            key="user_question"
        )

        if st.button("Consultar documento"):
            if not user_question.strip():
                st.warning("Escribe una pregunta antes de continuar.")
            else:
                with st.spinner("Analizando el documento..."):

                    # Recuperar fragmentos relevantes
                    docs = knowledge_base.similarity_search(
                        user_question,
                        k=4
                    )

                    # Modelo de chat
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        api_key=ke
                    )

                    # Preparar el contexto
                    contexto = "\n\n".join(
                        f"Página {doc.metadata.get('pagina', 'N/D')}:\n"
                        f"{doc.page_content}"
                        for doc in docs
                    )

                    # Generar respuesta
                    messages = [
                        (
                            "system",
                            "Responde en español usando únicamente "
                            "la información del contexto proporcionado. "
                            "Si la respuesta no aparece en el documento, "
                            "indica claramente que no encontraste "
                            "esa información. No inventes datos.\n\n"
                            f"CONTEXTO:\n{contexto}"
                        ),
                        ("human", user_question)
                    ]

                    response = llm.invoke(messages)

                st.markdown("### Respuesta:")
                st.markdown(response.content)

                with st.expander("Ver fragmentos consultados"):
                    for doc in docs:
                        st.caption(
                            f"Página {doc.metadata.get('pagina', 'N/D')}"
                        )
                        st.write(doc.page_content)
                        st.divider()

    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
```

