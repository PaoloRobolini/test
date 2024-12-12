import cv2
import mediapipe as mp

# Inizializza MediaPipe e OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# Funzione per rilevare il gesto del palmo aperto
def is_open_hand(landmarks):
    # Controlliamo la distanza tra il polso e le dita per determinare se la mano è aperta
    # La mano è considerata aperta se tutte le dita sono ben separate.

    # I punti di riferimento delle dita sono dati dai landmarks
    # Indici delle dita da verificare (0 è il polso, 1 è il pollice, 5 è il mignolo, ecc.)
    thumb_tip = landmarks[4]  # Pollice
    index_tip = landmarks[8]  # Indice
    middle_tip = landmarks[12]  # Medio
    ring_tip = landmarks[16]  # Anulare
    pinky_tip = landmarks[20]  # Mignolo

    # Distanza tra il pollice e l'indice (vediamo se è sufficientemente separato)
    thumb_index_dist = cv2.norm((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y), cv2.NORM_L2)
    index_middle_dist = cv2.norm((index_tip.x, index_tip.y), (middle_tip.x, middle_tip.y), cv2.NORM_L2)
    middle_ring_dist = cv2.norm((middle_tip.x, middle_tip.y), (ring_tip.x, ring_tip.y), cv2.NORM_L2)
    ring_pinky_dist = cv2.norm((ring_tip.x, ring_tip.y), (pinky_tip.x, pinky_tip.y), cv2.NORM_L2)

    # Se tutte le distanze tra le dita sono superiori a una certa soglia, consideriamo la mano aperta
    threshold = 0.05  # Può essere modificato in base alla tua applicazione
    if (thumb_index_dist > threshold and
            index_middle_dist > threshold and
            middle_ring_dist > threshold and
            ring_pinky_dist > threshold):
        return True
    return False


# Inizializza la videocamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte il frame in RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Esegui il rilevamento delle mani
    results = hands.process(frame_rgb)

    # Disegna i punti di riferimento e verifica se la mano è aperta
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Verifica se il gesto è un palmo aperto
            if is_open_hand(landmarks.landmark):
                cv2.putText(frame, 'Palm Open', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Disegna le mani sulla scena
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostra il frame
    cv2.imshow("Palm Gesture Detection", frame)

    # Esci se premi la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la videocamera e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
