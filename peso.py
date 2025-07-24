peso_anterior = 0  #Indica que la canasta está vacía

def deteccionCambioDePeso(peso_nuevo): # Se recibe el peso actual de la canasta
    si peso_nuevo > peso_anterior      # Si el peso nuevo es mayor que el anterior
        decir: "Aumento de peso"       # Decimos que hay un aumento en el peso
    
    peso_anterior = peso_nuevo         # Actualizamos nuestra variable de comparacion
                                       # El peso anterior ahora vale el peso que es mayor