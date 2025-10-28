import streamlit as st
from langchain.chat_models import ChatOpenAI
from PIL import Image
import os

st.set_page_config(
    page_title="Chatbot usando Langchain, OpenAI y Streamlit",
    page_icon="https://python.langchain.com/img/favicon.ico"
)

with st.sidebar:
    st.title("Usando la API de OpenAI con Streamlit y Langchain")

    model = st.selectbox('Elige el modelo', (
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k',
        'gpt-4'
    ), key="model")

    image = Image.open('streamlitcloud_chatbot/logos.png')#logos.png
    st.image(image, caption='OpenAI, Langchain y Streamlit')

    st.markdown("Integrando OpenAI con Streamlit y Langchain.")

    st.sidebar.button('Limpiar historial de chat', on_click=lambda: st.session_state.update(
        messages=[{"role": "assistant", "content": msg_chatbot}]
    ))


#def clear_chat_history():
#    st.session_state.messages = [{"role" : "assistant", "content": msg_chatbot}]

#openai_api_key = st.sidebar.text_input("Ingrese tu API Key de OpenAI y dale Enter para habilitar el chatbot", key = "chatbot_api_key", type = "password")
#st.sidebar.button('Limpiar historial de chat', on_click = clear_chat_history)


# Mensaje de bienvenida
msg_chatbot = """
        Soy un chatbot que está integrado a la API de OpenAI: 

        ### Preguntas frecuentes
        
        - ¿Quién eres?
        - ¿Cómo funcionas?
        - ¿Cuál es tu capacidad o límite de conocimientos?
        - ¿Puedes ayudarme con mi tarea/trabajo/estudio?
        - ¿Tienes emociones o conciencia?
        - Lo que desees
"""


# Obtener la API Key desde variable de entorno o secrets
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", None))
# openai_api_key = 

if not openai_api_key:
    st.error("❌ No se encontró la API Key. Define OPENAI_API_KEY como variable de entorno o en .streamlit/secrets.toml.")
    st.stop()

def get_response_openai(prompt, model):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=model,
        temperature=0
    )
    return llm.predict(prompt)

# Inicializar conversación
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": msg_chatbot}]

# Mostrar el historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capturar entrada
prompt = st.chat_input("Ingresa tu pregunta")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Si el último mensaje es del usuario, genera respuesta
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Esperando respuesta, dame unos segundos..."):
            response = get_response_openai(prompt, model)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
