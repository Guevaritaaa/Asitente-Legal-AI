from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_pagina_principal():
    respuesta_ia = client.get("/")
    assert respuesta_ia.status_code == 200
    assert respuesta_ia.headers["content-type"]

def test_guardrail():
    respuesta_ia = client.post(
        "api/chat",
        data={
            "pregunta": "Oye, me puedes escribir una funcion en Python que sume los numeros del 1 al 10"
        }
    )

    assert respuesta_ia.status_code == 200
    respuesta_json = respuesta_ia.json()

    assert "exclusivo para la gestión de devoluciones" in respuesta_json["respuesta"].lower()