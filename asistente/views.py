from django.shortcuts import render
from django.http import JsonResponse
from .motor_ia import motor  # Importamos el cerebro que construimos

def chat_view(request):

    if request.method == 'POST':
        pregunta = request.POST.get('pregunta')
        
        respuesta_ia = motor.consultar(pregunta)
        
        return JsonResponse(respuesta_ia)
        
    return render(request, 'asistente/chat.html')