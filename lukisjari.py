import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

# Palet warna
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
current_color = (0, 0, 0)  # Warna default hitam
canvas = np.zeros((480, 640, 3), dtype="uint8")  # Kanvas untuk menggambar

# Variabel untuk melacak posisi jari sebelumnya agar garis mulus
prev_x, prev_y = None, None

def select_color(x, y):
    """Pilih warna berdasarkan koordinat di palet."""
    global current_color
    if 10 <= x <= 60:  # Area palet warna
        for i, color in enumerate(colors):
            if 10 + i * 50 <= y <= 60 + i * 50:
                current_color = color

def is_fingers_closed(landmarks):
    """Deteksi jika semua jari tertutup (reset)."""
    return all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])

def is_three_fingers_open(landmarks):
    """Deteksi jika tiga jari terbuka (hapus)."""
    return (landmarks[8].y < landmarks[6].y and  # Telunjuk terbuka
            landmarks[12].y < landmarks[10].y and  # Jari tengah terbuka
            landmarks[16].y < landmarks[14].y)  # Jari manis terbuka

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membuka kamera")
            break

        # Balik frame agar tidak mirror
        frame = cv2.flip(frame, 1)  # 1 untuk membalik horizontal

        # Konversi frame ke RGB (MediaPipe membutuhkan input RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Jika ada tangan terdeteksi
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                # Koordinat ujung telunjuk dan ibu jari
                x, y = int(landmarks[8].x * 640), int(landmarks[8].y * 480)

                # Pilih warna jika telunjuk terbuka dan tengah tertutup
                if landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y:
                    select_color(x, y)

                # Reset jika semua jari tertutup
                if is_fingers_closed(landmarks):
                    canvas = np.zeros((480, 640, 3), dtype="uint8")

                # Hapus jika tiga jari terbuka
                elif is_three_fingers_open(landmarks):
                    cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)

                # Lukis dengan telunjuk dan ibu jari terbuka
                elif landmarks[8].y < landmarks[6].y and landmarks[4].x < landmarks[3].x:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 5)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None  # Reset posisi

        # Tampilkan palet warna
        for i, color in enumerate(colors):
            cv2.rectangle(frame, (10, 10 + i * 50), (60, 60 + i * 50), color, -1)

        # Gabungkan kanvas dan frame kamera
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Tampilkan frame
        cv2.imshow("Melukis dengan Gerakan Tangan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
