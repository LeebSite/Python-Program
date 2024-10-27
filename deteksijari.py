import cv2
import mediapipe as mp

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membuka kamera")
            break

        # Konversi frame ke RGB (MediaPipe membutuhkan input RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Jika ada tangan terdeteksi
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # List koordinat landmark jari
                landmarks = hand_landmarks.landmark

                # Logika untuk menghitung jari yang terbuka (thumb excluded)
                fingers = [4, 8, 12, 16, 20]  # Landmark ujung jari

                # Deteksi jari yang terbuka berdasarkan posisi y
                open_fingers = 0
                for i in range(1, 5):
                    if landmarks[fingers[i]].y < landmarks[fingers[i] - 2].y:
                        open_fingers += 1

                # Tambahkan logika untuk ibu jari
                thumb_open = landmarks[fingers[0]].x < landmarks[fingers[0] - 1].x
                open_fingers += 1 if thumb_open else 0

                # Tampilkan angka yang terdeteksi
                cv2.putText(frame, f"Angka Tangan: {open_fingers}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Tampilkan frame
        cv2.imshow("Deteksi Angka Tangan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
