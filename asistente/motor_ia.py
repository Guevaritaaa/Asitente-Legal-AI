from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
import os

class AsistenteLegal:
    def __init__(self):
        print("Configurando modelos locales...")
        Settings.llm = Ollama(
            model="llama3.1",
            request_timeout=120.0,
            context_window=2048,
            system_prompt=(
                "Eres el Asistente Virtual de Devoluciones de 'GuevaraStore'. "
                "Tu objetivo es ayudar a los clientes a saber si su producto aplica para una devolución "
                "basándote exclusivamente en la política de la empresa. "
                "Sé amable, empático pero muy preciso con las reglas. "
                "Si el cliente no cumple un requisito, explícale de forma cordial por qué no procede."
            )
        )
        
        modelo_embeddings = OllamaEmbedding(model_name="nomic-embed-text")
        Settings.embed_model = modelo_embeddings

        separador_semantico = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=modelo_embeddings
        )

        ruta_datos = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datos')
        
        print("Leyendo los PDFs con PyMuPDF...")
        extractor_pdf = {".pdf": PyMuPDFReader()}
        documentos = SimpleDirectoryReader(ruta_datos, file_extractor=extractor_pdf).load_data()

        print("Procesando cortes semánticos... (Esto tomará un poco más de tiempo la primera vez)")
        
        nodos = separador_semantico.get_nodes_from_documents(documentos)

        print("Vectorizando los nuevos chunks...")
        self.indice = VectorStoreIndex(nodos)
        self.motor_preguntas = self.indice.as_query_engine(similarity_top_k=5)

    def consultar(self, pregunta):
        respuesta = self.motor_preguntas.query(pregunta)
        
        texto_limpio = str(respuesta).replace("**Rewrite**", "").replace("**Repeat**", "").strip()
        fuentes = [n.text for n in respuesta.source_nodes]
        
        return {
            "respuesta": texto_limpio,
            "fuentes": fuentes
        }

motor = AsistenteLegal()