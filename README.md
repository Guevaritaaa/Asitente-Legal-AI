# Asistente de Devoluciones AI - GuevaraStore

**Demo en Vivo:** [Prueba el Asistente Aquí](https://api-guevarastore.onrender.com)

Un backend inteligente construido con **FastAPI** y una arquitectura **RAG Multi-Modelo**, diseñado para auditar y gestionar solicitudes de devolución en comercio electrónico basándose estrictamente en políticas corporativas (política ficticia para fines prácticos).

## Arquitectura Técnica
* **Framework:** FastAPI (Python)
* **Orquestador AI:** LlamaIndex
* **Cerebro Generativo (LLM):** Groq (Llama-3.3-70B)
* **Motor de Embeddings:** Voyage AI (voyage-4)
* **Procesamiento de Documentos:** Chunking Semántico (PyMuPDF)

## Características Principales
* **RAG Multi-Modelo:** Desacoplamiento de embeddings y generación para optimizar latencia y esquivar cuotas estrictas de API.
* **Prompt Engineering Avanzado:** Implementación de *Guardrails* de alta prioridad para evitar alucinaciones, inyecciones de prompt (Prompt Injection) y garantizar el rechazo inmediato de artículos no retornables (ej. caducidad, higiene, liquidación).
* **Chunking Semántico:** División inteligente del documento de políticas evaluando el cambio de contexto real de las oraciones, no solo el conteo de palabras.
* **Testing Automatizado:** Suite de pruebas de integración y seguridad implementada con `pytest` para garantizar la estabilidad del servidor y validar la eficacia de los *guardrails* de la IA.

## Cómo ejecutar en local
Si deseas clonar y probar este proyecto en tu propia máquina:

1. Clona el repositorio e instala las dependencias:
   ```bash
   pip install -r requirements.txt
   
Crea un archivo .env basándote en el .env.example e inserta tus API Keys de Groq y Voyage AI.

Inicia el servidor con este comando:

En la Terminal escribe: 

fastapi dev main.py

Abre http://localhost:8000 en tu navegador.

🚀 Roadmap (Próximas Mejoras)
[ ] Persistencia Vectorial (PostgreSQL + pgvector): Migración de los embeddings a una base de datos para evitar la re-indexación en cada arranque del servidor, reduciendo a cero el consumo redundante de la API de Voyage AI.

[ ] Memoria de Conversación: Implementación de historial de chat para que el LLM recuerde el contexto inmediato de la sesión actual.