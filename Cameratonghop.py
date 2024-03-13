import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Khởi tạo Mediapipe Solutions
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Chỉ số landmark được chọn cho cả hai mắt
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]

# Tạo đối tượng kết nối với camera
cap = cv2.VideoCapture(0)

# Khởi tạo danh sách để lưu trữ tọa độ của mắt
left_eye_coordinates = []
right_eye_coordinates = []

with mp_facemesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Chuyển đổi màu sắc của khung hình
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Xử lý khung hình
        results = face_mesh.process(rgb_frame)

        # Xử lý từng khuôn mặt trong khung hình
        if results.multi_face_landmarks:
            for facial_landmarks in results.multi_face_landmarks:
                # Thu thập tọa độ của mắt trái và mắt phải
                left_eye_lm = [facial_landmarks.landmark[idx] for idx in chosen_left_eye_idxs]
                right_eye_lm = [facial_landmarks.landmark[idx] for idx in chosen_right_eye_idxs]

                # Tính trung bình tọa độ cho mỗi mắt
                left_eye_avg = np.mean([[lm.x, lm.y] for lm in left_eye_lm], axis=0)
                right_eye_avg = np.mean([[lm.x, lm.y] for lm in right_eye_lm], axis=0)

                left_eye_coordinates.append(left_eye_avg)
                right_eye_coordinates.append(right_eye_avg)

                # Vẽ landmark lên khuôn mặt
                mp_drawing.draw_landmarks(
                    frame,
                    facial_landmarks,
                    mp_facemesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # Hiển thị khung hình
        cv2.imshow('Face Mesh', frame)

        # Thoát khỏi vòng lặp khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()

# Kiểm tra dữ liệu trước khi vẽ biểu đồ
if left_eye_coordinates and right_eye_coordinates:
    left_eye_coordinates = np.array(left_eye_coordinates)
    right_eye_coordinates = np.array(right_eye_coordinates)

    plt.figure(figsize=(10, 4))
    plt.plot(left_eye_coordinates[:, 0], label='Left Eye X')
    plt.plot(left_eye_coordinates[:, 1], label='Left Eye Y')
    plt.plot(right_eye_coordinates[:, 0], label='Right Eye X')
    plt.plot(right_eye_coordinates[:, 1], label='Right Eye Y')
    plt.title('Eye Coordinates Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Coordinate')
    plt.legend()
    plt.show()
else:
    print("No data.")
