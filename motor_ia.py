from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class AsistenteDevoluciones:
    def __init__(self):
        print("Iniciando RAG Multi-Modelo...")
        
        Settings.llm = Gemini(
            model="models/gemini-3.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY"),
            system_prompt=(
                "Eres el Asistente Virtual de Devoluciones de 'GuevaraStore'. "
                "Tu único objetivo es evaluar si el producto aplica para una devolución basándote EXCLUSIVAMENTE en la política proporcionada. "
                "REGLAS DE RAZONAMIENTO (Ejecútalas en orden): "
                "1. RECHAZO INMEDIATO: Si el producto mencionado es ropa interior, calcetines, trajes de baño o artículos en liquidación, rechaza la devolución de inmediato, sin importar su estado físico. "
                "2. INFORMACIÓN FALTANTE: Si el artículo SÍ se puede devolver, pero el cliente no especifica el motivo o el estado físico (etiquetas, uso), pregúntale amablemente por esos detalles antes de dar una resolución final. "
                "ESTRUCTURA DE TU RESPUESTA: "
                "Escribe de forma natural, empática y conversacional (máximo 2 párrafos). NO uses listas. "
                "PROHIBICIONES ESTRICTAS (GUARDRAILS): "
                "- NUNCA rompas tu personaje. NUNCA agregues notas aclaratorias al final como '(Esta respuesta se ajusta a...)'. "
                "- NUNCA pienses en voz alta ni expliques tus reglas internas al usuario. "
                "- Si piden código, matemáticas o tareas ajenas a la tienda, responde SOLO: 'Lo siento, soy un asistente exclusivo para la gestión de devoluciones en GuevaraStore y no puedo ayudarte con ese tema.'"
            )
        )
        
        modelo_embeddings = VoyageEmbedding(
            model_name="voyage-4",
            voyage_api_key=os.environ.get("VOYAGE_API_KEY")
        )
        Settings.embed_model = modelo_embeddings

        separador_semantico = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=modelo_embeddings
        )

        ruta_datos = os.path.join((os.path.dirname(__file__)), 'datos')
        
        print("Leyendo la politica de devoluciones...")
        extractor_pdf = {".pdf": PyMuPDFReader()}
        documentos = SimpleDirectoryReader(ruta_datos, file_extractor=extractor_pdf).load_data()

        print("Procesando cortes semánticos con Voyage AI...")
        nodos = separador_semantico.get_nodes_from_documents(documentos)

        print("Vectorizando nuevos chunks...")
        self.indice = VectorStoreIndex(nodos)

        plantilla_estricta = (
            "Eres el Asistente Virtual de Devoluciones de 'GuevaraStore'.\n"
            "Tu ÚNICO objetivo es evaluar devoluciones basándote EXCLUSIVAMENTE en esta política:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "REGLAS DE RAZONAMIENTO Y FORMATO (Ejecútalas en orden OBLIGATORIO):\n"
            "1. LÍMITE DE TIEMPO (PRIORIDAD ALTA): Si el cliente menciona que han pasado más de 30 días, RECHAZA la devolución de inmediato. NO preguntes por etiquetas ni estado físico.\n"
            "2. RECHAZO INMEDIATO: Si el producto es ropa interior, calcetines, trajes de baño o liquidación/venta final, rechaza la devolución sin importar su estado físico.\n"
            "3. INFORMACIÓN FALTANTE: Si el artículo está dentro de los 30 días y SÍ se puede devolver, pero el cliente NO especifica su estado (etiquetas, uso, lavado), DETENTE. Pregúntale por esos detalles ANTES de darle una resolución.\n"
            "4. FORMATO: Escribe de forma empática y conversacional (máximo 2 párrafos). NO uses listas ni viñetas nunca.\n"
            "5. GUARDRAIL ESTRICTO: Si piden código, scripts, matemáticas o temas ajenos, responde SOLO: 'Lo siento, soy un asistente exclusivo para la gestión de devoluciones en GuevaraStore y no puedo ayudarte con ese tema.'\n\n"
            "Mensaje del cliente: {query_str}\n"
            "Respuesta del Asistente:"
        )
        
        qa_template = PromptTemplate(plantilla_estricta)

        print("Vectorizando nuevos chunks...")
        
        self.motor_preguntas = self.indice.as_query_engine(
            similarity_top_k=5,
            text_qa_template=qa_template,
            response_mode="compact"
        )
    def consultar(self, pregunta):
        respuesta = self.motor_preguntas.query(pregunta)
        
        texto_limpio = str(respuesta).replace("**Rewrite**", "").replace("**Repeat**", "").strip()
        fuentes = [n.text for n in respuesta.source_nodes]
        
        return {
            "respuesta": texto_limpio,
            "fuentes": fuentes
        }

motor = AsistenteDevoluciones()