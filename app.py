import cv2
from deepface import DeepFace

emociones_es = {
    "happy": "Feliz",
    "sad": "Triste",
    "angry": "Enojado",
    "surprise": "Sorprendido",
    "fear": "Miedo",
    "disgust": "Asco",
    "neutral": "Neutral"
}

# Abrir webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Analizar Emociones 
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emocion = result[0]['dominant_emotion']
        emocion_es = emociones_es.get(emocion, emocion)  # Traducir o dejar como está si no se encuentra
        cv2.putText(frame, f'Emoción: {emocion_es}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    except Exception as e:
        print("Error analizando emociones:", e)

    cv2.imshow('Detector de emociones', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
