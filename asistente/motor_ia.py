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
        
        # Inicializar Modelo Embendings
        modelo_embeddings = OllamaEmbedding(model_name="nomic-embed-text")
        Settings.embed_model = modelo_embeddings

        # Configurar el Separador Semántico
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