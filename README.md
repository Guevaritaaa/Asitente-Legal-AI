# Asistente de Devoluciones AI - GuevaraStore

Un backend inteligente construido con **FastAPI** y una arquitectura **RAG Multi-Modelo**, diseñado para auditar y gestionar solicitudes de devolución en comercio electrónico basándose estrictamente en políticas corporativas(para fines practicos se hizo uso de una politica ficticia).

## Arquitectura Técnica
* **Framework:** FastAPI (Python)
* **Orquestador AI:** LlamaIndex
* **Cerebro Generativo (LLM):** Google Gemini 3.5 Flash
* **Motor de Embeddings:** Voyage AI (voyage-4)
* **Procesamiento de Documentos:** Chunking Semántico (PyMuPDF)

## Características Principales
* **RAG Multi-Modelo:** Desacoplamiento de embeddings y generación para optimizar latencia y esquivar cuotas estrictas de API.
* **Prompt Engineering Avanzado:** Implementación de *Guardrails* de alta prioridad para evitar alucinaciones, inyecciones de prompt (Prompt Injection) y garantizar el rechazo inmediato de artículos no retornables (ej. caducidad, higiene, liquidación).
* **Chunking Semántico:** División inteligente del documento de políticas evaluando el cambio de contexto real de las oraciones, no solo el conteo de palabras.

## Despliegue
Este proyecto está configurado para ser desplegado en servicios cloud como Render o Railway mediante la inyección segura de variables de entorno (`GOOGLE_API_KEY`, `VOYAGE_API_KEY`).

## Roadmap (Próximas Mejoras)
- [ ] **Persistencia Vectorial (PostgreSQL + pgvector):** Migración de los embeddings a una base de datos para evitar la re-indexación en cada arranque del servidor, reduciendo a cero el consumo redundante de la API de Voyage AI.
- [ ] **Rediseño de UI/UX:** Mejoras visuales y de usabilidad en el frontend (`chat.html`) para ofrecer una experiencia más moderna y responsiva.
- [ ] **Memoria de Conversación:** Implementación de historial de chat para que el LLM recuerde el contexto inmediato de la sesión actual.
