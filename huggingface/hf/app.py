import os
import streamlit as st
from PIL import Image

# LangChain (nueva organización de paquetes)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# =====================
# Configuración general
# =====================
st.set_page_config(page_title="Chatbot usando ChatGPT", page_icon="⚽")

# Toma las claves de variables de entorno o de st.secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")

# Exporta para los SDKs que las leen desde env
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Nombre del índice existente en Pinecone (ya creado y poblado)
INDEX_NAME = "knowledge-base-eliminatorias"

# ====================
# Mensaje de bienvenida
# ====================
MSG_CHATBOT = """
Soy un chatbot que te ayudará a conocer sobre las eliminatorias sudamericanas: 
### Puedo ayudarte con las siguientes preguntas:
- ¿Quién es el líder en la tabla de posiciones?
- ¿Cuáles son los próximos partidos de Perú?
- Bríndame la tabla de posiciones
- Y muchas más...
"""

# ======================
# Reinicio de historial
# ======================
if st.session_state.get("clear_chat", False):
    st.session_state.clear_chat = False
    st.session_state.messages = [{"role": "assistant", "content": MSG_CHATBOT}]
    st.rerun()

# =========
# Sidebar
# =========
with st.sidebar:
    st.title("Chatbot usando OpenAI (ChatGPT)")
    try:
        image = Image.open("huggingface/hf/conmebol.jpg")
        st.image(image, caption="Conmebol")
    except Exception:
        st.info("Sube la imagen en huggingface/hf/conmebol.jpg (opcional)")
    st.markdown(
        """
        ### Propósito
        Este chatbot utiliza una base de conocimiento (Pinecone) con información del sitio web de Marca.
        Usa LangChain con ChatGPT de OpenAI.
        ### Fuentes 
        - Marca - (https://www.marca.com)
        """
    )
    if st.button("🧹 Limpiar chat"):
        st.session_state.clear_chat = True
        st.rerun()

# =======================
# Inicializar historial
# =======================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": MSG_CHATBOT}]

# =======================
# Mostrar historial
# =======================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ====================
# Generar respuesta
# ====================
def generate_openai_pinecone_response(user_query: str) -> str:
    # Modelo: puedes cambiar a "gpt-4o-mini" (barato/rápido) o "gpt-4o" (mejor calidad)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", #gpt-4o-mini
        temperature=0.85,
    )

    template = """Responde a la pregunta basada en el siguiente contexto.
Si no puedes responder, di: "No lo sé, disculpa, puedes buscar en internet."

Contexto:
{context}

Pregunta: {question}

Respuesta (usa también emoticones si suma claridad):
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    embeddings = OpenAIEmbeddings()

    # Con el SDK 3.x de Pinecone NO necesitas "environment".
    # Solo asegúrate de que tu índice INDEX_NAME exista en el mismo proyecto de tu API key.
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        chain_type_kwargs={"prompt": prompt}
    )

    # En LC 0.2+ usa invoke({"query": ...})
    out = qa.invoke({"query": user_query})
    # El dict suele traer "result" y "source_documents"
    # return out["result"] if isinstance(out, dict) and "result" in out else str(out)
    return out["result"]

# ====================
# Interfaz principal
# ====================
prompt = st.chat_input("Ingresa tu pregunta")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Esperando respuesta..."):
            response = generate_openai_pinecone_response(st.session_state.messages[-1]["content"])
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
