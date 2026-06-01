\#Asistente RAG de Devoluciones (GuevaraStore)



Este proyecto es un asistente virtual diseñado para automatizar la atención al cliente en procesos de devolución, utilizando una arquitectura RAG 100% local.



\*\*Actualización Reciente:\*\* El backend fue migrado de Django a microservicios con \*\*FastAPI\*\* para optimizar el ciclo de petición-respuesta y preparar la infraestructura para un futuro despliegue moderno.



\## Stack Tecnológico

\* \*\*Backend:\*\* FastAPI (Python), Uvicorn.

\* \*\*Inteligencia Artificial:\*\* LlamaIndex, Ollama (Llama 3.1).

\* \*\*Embeddings \& Procesamiento:\*\* nomic-embed-text, Semantic Chunking (PyMuPDF).

\* \*\*Frontend:\*\* HTML5, CSS3, JavaScript.



\##Arquitectura y Flujo de Datos

1\. \*\*Ingesta de Documentos:\*\* El sistema lee la política de devoluciones corporativa en PDF.

2\. \*\*Fragmentación Semántica:\*\* Se divide el texto inteligentemente para no perder contexto.

3\. \*\*Vectorización:\*\* Los fragmentos se convierten en embeddings para realizar búsquedas por similitud matemática.

4\. \*\*Guardrails (Prompt Engineering):\*\* Llama 3.1 evalúa la política, rechaza artículos no retornables (ej. ropa interior) y evita alucinaciones o desviaciones del tema.

