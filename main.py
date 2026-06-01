from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from motor_ia import motor

app = FastAPI()

@app.get("/")
def pagina_principal():
    return FileResponse("chat.html")

@app.post("/api/chat")
async def responderChat(pregunta: str = Form(...)):
    respuesta_ia = motor.consultar(pregunta)
    return respuesta_ia