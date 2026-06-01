import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
clave = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=clave)

print("Buscando modelos VECTORIZADORES (Embeddings) disponibles...\n")

for modelo in genai.list_models():
    # Ahora filtramos buscando específicamente la capacidad de hacer embeddings
    if 'embedContent' in modelo.supported_generation_methods:
        print(f"Nombre exacto: {modelo.name}")