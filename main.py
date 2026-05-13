# cv2 - OpenCV - reconhecimento facial
import cv2  
import mediapipe as mp  
import time
import subprocess
from pathlib import Path

# usando windows entao os
import os 

def play_video(video_path: Path) -> None:
    # abre o vídeo no player padrão
    os.startfile(str(video_path))

def close_video(video_path: Path) -> None:
    # fechar players
    players = ["Video.UI.exe", "vlc.exe", "wmplayer.exe", "mpc-hc64.exe"]
    for player in players:
        subprocess.run(
            ["taskkill", "/F", "/IM", player],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

# desenhar janela
def draw_warning(frame, text="tururu"):
    h, w = frame.shape[:2]
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (80, 255, 160) , 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160) , 2)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    
def main():
    # limiar de tristeza
    sad_threshold = 0.02 
    timer = 1.0 # tempo sendo triste
    
    sad_naruto = Path("./assets/sad.mp4").resolve()
    if not sad_naruto.exists():
        print("Could not find sad.mp4 :(")
        return
    
    #
    face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    cam = cv2.VideoCapture(0)
    
    sadness = None
    video_playing = False

    while True:
        ret, frame = cam.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = face_mesh_landmarks.process(rgb_frame)
        face_landmark_points = processed_image.multi_face_landmarks

        current = time.time()

        if face_landmark_points:
            landmarks = face_landmark_points[0].landmark
            
            landmark_points = [61, 291, 0]
            for x in landmark_points:
                ponto = landmarks[x]
                cx, cy = int(ponto.x * width), int(ponto.y * height)
                # cv2.circle(imagem, centro, raio, cor(BGR), espessura)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            # PONTOS CHAVE PARA TRISTEZA
            # 61, 291: Cantos da boca
            # 0, 17: Centro do lábio superior e inferior
            # 52, 282: Topo das sobrancelhas
            
            boca_canto_esq = landmarks[61]
            boca_canto_dir = landmarks[291]
            labio_superior = landmarks[0]
            
            # Cálculo simples: se os cantos da boca estiverem muito abaixo do centro do lábio
            # calculamos a média da altura dos cantos e comparamos com o lábio superior
            boca_media_y = (boca_canto_esq.y + boca_canto_dir.y) / 2
            tristeza_score = boca_media_y - labio_superior.y

            if tristeza_score > sad_threshold:
                if sadness is None:
                    sadness = current
                
                if (current - sadness) >= timer:
                    if not video_playing:
                        play_video(sad_naruto)
                        video_playing = True
            else:
                # reseta se ficar feliz
                sadness = None
                if video_playing and tristeza_score < (sad_threshold - 0.02):
                    close_video(sad_naruto)
                    video_playing = False
        else:
            sadness = None

        cv2.imshow('tururuu', frame)
        if cv2.waitKey(1) == 27: break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()