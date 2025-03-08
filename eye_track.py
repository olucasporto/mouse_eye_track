import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Captura da webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()  # Tamanho da tela

# Filtros para suavizar movimento do mouse
prev_x, prev_y = 0, 0
alpha = 0.5  # Fator de suavização

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Inverte a imagem para parecer um espelho
    frame = cv2.flip(frame, 1)

    # Converte BGR para RGB (necessário para o MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a face para detectar os olhos
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            h, w, _ = frame.shape  # Dimensões do frame

            # Pegamos os pontos ao redor da pupila para melhorar a precisão
            left_eye_points = [469, 470, 471, 472]  # Olho esquerdo
            right_eye_points = [474, 475, 476, 477]  # Olho direito

            # Média dos pontos da pupila para evitar ruído
            left_eye_x = np.mean([face_landmarks.landmark[i].x for i in left_eye_points])
            left_eye_y = np.mean([face_landmarks.landmark[i].y for i in left_eye_points])

            right_eye_x = np.mean([face_landmarks.landmark[i].x for i in right_eye_points])
            right_eye_y = np.mean([face_landmarks.landmark[i].y for i in right_eye_points])

            # Centro dos olhos
            eye_x = (left_eye_x + right_eye_x) / 2
            eye_y = (left_eye_y + right_eye_y) / 2

            # Converte posição dos olhos para a tela do computador
            mouse_x = int(eye_x * screen_w)
            mouse_y = int(eye_y * screen_h)

            # Aplica suavização para evitar tremores bruscos
            smooth_x = int(prev_x * alpha + mouse_x * (1 - alpha))
            smooth_y = int(prev_y * alpha + mouse_y * (1 - alpha))

            # Move o mouse
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.1)

            # Atualiza os valores anteriores para suavização
            prev_x, prev_y = smooth_x, smooth_y

            # Desenha os pontos dos olhos na imagem
            for i in left_eye_points + right_eye_points:
                lx, ly = int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)
                cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

    # Exibe a câmera
    cv2.imshow("Eye Tracking Melhorado", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
