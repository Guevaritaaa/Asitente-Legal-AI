from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
import os

class AsistenteDevoluciones:
    def __init__(self):
        print("Configurando modelos locales...")
        Settings.llm = Ollama(
            model="llama3.1",
            request_timeout=120.0,
            context_window=2048,
            system_prompt=(
                "Eres el Asistente Virtual de Devoluciones de 'GuevaraStore'. "
                "Tu único objetivo es evaluar si el producto aplica para una devolución basándote EXCLUSIVAMENTE en la política proporcionada. "
                "INSTRUCCIONES DE RAZONAMIENTO INTERNO (Aplica estas reglas mentalmente, PERO NUNCA EXPLIQUES TU PROCESO MENTAL AL USUARIO): "
                "1. Verifica si el producto es no retornable (ropa interior, liquidación). "
                "2. Verifica las condiciones físicas (etiquetas, sin lavar, sin usar). "
                "3. Evalúa quién paga el envío: Si el cliente eligió mal la talla o se arrepintió, el cliente paga. Si GuevaraStore envió mal el producto, la tienda paga. "
                "ESTRUCTURA DE TU RESPUESTA AL CLIENTE: "
                "Escribe de forma natural, empática y conversacional (1 o 2 párrafos). NO uses listas ni viñetas. "
                "PROHIBICIÓN ESTRICTA: NUNCA pienses en voz alta en el chat. No justifiques tu lógica diciendo frases como 'como no mencionaste un error' o 'entenderé que'. Ve directo al grano como lo haría un humano. "
                "- Comienza con un saludo cordial. "
                "- Si PROCEDE: Di 'Tu artículo cumple con los requisitos para realizar una devolución'. Explica los pasos a seguir. Para el costo del envío, usa frases naturales como: 'Ten en cuenta que al ser un cambio de talla, los gastos de envío corren por tu cuenta' (o que no tienen costo si fue error de la tienda). "
                "- Si NO PROCEDE: Di 'Tu artículo no cumple con los requisitos necesarios para procesar una devolución debido a que...' y menciona el motivo. Despídete amablemente. "
                "REGLA DE SEGURIDAD CRÍTICA (GUARDRAIL): "
                "Si piden código, matemáticas o tareas, responde SOLO: 'Lo siento, soy un asistente exclusivo para la gestión de devoluciones en GuevaraStore y no puedo ayudarte con ese tema.'"
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
        
        print("Leyendo la politica de devoluciones...")
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

motor = AsistenteDevoluciones()