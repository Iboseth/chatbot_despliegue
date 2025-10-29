import streamlit as st
from PIL import Image
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  # nombre actual
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ========================
# Configuración de claves
# ========================
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = "us-east-1"
index_name = "knowledge-base-eliminatorias"

# =====================
# Configuración de app
# =====================
st.set_page_config(page_title="Chatbot usando ChatGPT", page_icon="⚽")

# ====================
# Mensaje de bienvenida
# ====================
msg_chatbot = """
Soy un chatbot que te ayudará a conocer sobre las eliminatorias sudamericanas: 
### Puedo ayudarte con las siguientes preguntas:
- ¿Quién es el líder en la tabla de posiciones?
- ¿Cuáles son los próximos partidos de Perú?
- Bríndame la tabla de posiciones
- Y muchas más...
"""

# ======================
# Reinicio de historial (antes de dibujar mensajes)
# ======================
if st.session_state.get("clear_chat", False):
    st.session_state.clear_chat = False
    st.session_state.messages = [{"role": "assistant", "content": msg_chatbot}]
    st.rerun()

# =========
# Sidebar
# =========
with st.sidebar:
    st.title("Chatbot usando OpenAI (ChatGPT)")
    image = Image.open('huggingface/hf/conmebol.jpg')#src/conmebol.jpg
    st.image(image, caption='Conmebol')
    st.markdown("""
        ### Propósito
        Este chatbot utiliza una base de conocimiento (Pinecone) con información del sitio web de Marca.
        Usa Langchain con ChatGPT de OpenAI.
        ### Fuentes 
        - Marca - (https://www.marca.com)
    """)

    if st.button("🧹 Limpiar chat"):
        st.session_state.clear_chat = True
        st.rerun()

# =======================
# Inicializar historial si es necesario
# =======================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": msg_chatbot}]

# =======================
# Mostrar historial
# =======================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ====================
# Generar respuesta
# ====================
def generate_openai_pinecone_response(prompt_input):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.85
    )

    template = """Responde a la pregunta basada en el siguiente contexto.
    Si no puedes responder, di: "No lo sé, disculpa, puedes buscar en internet."
    Contexto:
    {context}
    Pregunta: {question}
    Respuesta usando también emoticones:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(),
        verbose=True,
        chain_type_kwargs={"prompt": prompt}
    )

    #return qa.run(prompt_input)
    result = qa.invoke({"query": prompt_input})
    return result if isinstance(result, str) else result.get("result", "")


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
            response = generate_openai_pinecone_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
