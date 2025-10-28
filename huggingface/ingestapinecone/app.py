import os
# import pinecone
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Configura la clave API de Pinecone en la variable de entorno
api_key = os.environ["PINECONE_API_KEY"]
os.environ["PINECONE_API_KEY"] = api_key

# Nombre del índice que deseas crear
index_name = "knowledge-base-eliminatorias"

# Dimension del modelo, ajustado para Stable Diffusion
dimension = 1536

# Especificaciones del servidor donde se alojará el índice
spec = ServerlessSpec(
    cloud="aws",  # Ajusta según la nube que estés utilizando, "aws", "gcp", etc.
    region="us-east-1"  # Asegúrate de que la región esté disponible en tu plan
)

# Inicializa el cliente de Pinecone
pinecone_client = Pinecone()

# Crea un índice en Pinecone
pinecone_client.create_index(
    name=index_name, 
    dimension=dimension,
    spec=spec  # Incluye las especificaciones del servidor
)

print("Index " + index_name + " creado con éxito en Pinecone")

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

loader = WebBaseLoader(
    [
    "https://www.marca.com/co/2023/10/17/652e070f22601d73648b4585.html", 
    "https://hiraoka.com.pe/blog/post/eliminatorias-sudamericanas-mundial-2026-calendario-partidos-y-fechas"
    ]
)

data = loader.load()

#Genera varios fragmentos de 400 tokens
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 400, 
    chunk_overlap = 20
)

docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_documents(docs, embeddings, index_name = index_name)

print("Se guardaron en total " + str(len(docs)) + " embedings en Pinecone (index : " + index_name + ")")