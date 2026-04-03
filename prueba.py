from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader

print("Configurando motores locales...")
Settings.llm = Ollama(
    model="llama3.1",
    request_timeout=120.0,
    context_window=2048,
system_prompt=(
            "Eres un asesor legal corporativo altamente capacitado, pero muy amable y pedagógico. "
            "Tu trabajo es explicar la Ley Federal de Protección de Datos Personales a personas que no saben nada de derecho. "
            "Usa SIEMPRE un lenguaje sencillo, claro y directo, organizando en viñetas si es necesario. "
            "Debes basar todas tus respuestas estrictamente en el documento proporcionado. "
            "REGLA LÓGICA CRÍTICA: Presta especial atención a las palabras 'exceptuados', 'no aplica' o 'excepciones'. "
            "Nunca confundas a las personas a las que NO les aplica la ley, con el objetivo principal de la ley. "
            "NUNCA copies y pegues artículos de la ley sin explicarlos. "
            "Si la respuesta no se encuentra en el texto, responde: 'No tengo esa información en el documento'. "
            "IMPORTANTE: NUNCA comiences tu respuesta con la palabra 'Rewrite' ni 'Repeat'."
        )
)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# 2. Leer los documentos
print("Leyendo los PDFs con PyMuPDF...")
extractor_pdf = {".pdf": PyMuPDFReader()}
documentos = SimpleDirectoryReader("datos", file_extractor=extractor_pdf).load_data()

# 3. Crear el índice (La base de datos vectorial en memoria)
print("Indexando y vectorizando (Esto puede tomar unos segundos)...")
indice = VectorStoreIndex.from_documents(documentos)

# 4. Crear el motor de búsqueda y consultar
print("¡Sistema listo! Realizando consulta...")
motor_preguntas = indice.as_query_engine(similarity_top_k=5)

while (True):

    pregunta = input("Que deseas preguntar?")
    
    if pregunta.lower() == "exit":
        break
    else:
        respuesta = motor_preguntas.query(pregunta)
        for n in respuesta.source_nodes:
            print(n)

        print("\n--- RESPUESTA BASADA EN EL MANUAL ---")
        print(respuesta)