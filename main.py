from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from motor_ia import motor

app = FastAPI()

@app.get("/")
def pagina_principal():
    return FileResponse("chat.html")

@app.post("/api/chat")
async def responderChat(pregunta: str = Form(...)):
    try:
        respuesta_ia = motor.consultar(pregunta)
        return respuesta_ia
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Ocurrió un error en el servidor: {str(e)}"
        )