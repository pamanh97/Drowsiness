import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import pygame

# Khởi tạo pygame và tải file âm thanh báo động
pygame.init()
default_alarm_sound_path = "C:/Laptrinh/Driver-Drowsiness-detection-using-Mediapipe-in-Python/audio/wake_up.wav"

def get_alarm_sound():
    alarm_sound_path = input(f"Enter path to alarm sound file (default: '{default_alarm_sound_path}'): ")
    if not alarm_sound_path:
        alarm_sound_path = default_alarm_sound_path
    return pygame.mixer.Sound(alarm_sound_path)

alarm_sound = get_alarm_sound()

def get_mediapipe_app(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

def distance(point_1, point_2):
    dist = np.linalg.norm(np.array(point_1) - np.array(point_2))
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    coords_points = []
    for i in refer_idxs:
        lm = landmarks[i]
        coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
        if coord:
            coords_points.append(coord)

    if len(coords_points) == 6:
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    else:
        ear = 0.0

    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0
    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)
    return frame

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

def adjust_image(frame):
    # Chuyển đổi sang không gian màu LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Áp dụng CLAHE cho l-channel để cải thiện tương phản
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Gộp các kênh và chuyển đổi trở lại không gian màu BGR
    adjusted_lab = cv2.merge([l, a, b])
    adjusted_frame = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    return adjusted_frame

def calculate_yawning_ratio(landmarks, frame_width, frame_height):
    # Chọn điểm mốc miệng
    mouth_idxs = [61, 146, 91, 181, 84, 17]
    mouth_coordinates = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in mouth_idxs]
    if None not in mouth_coordinates:
        # Tính toán tỷ lệ ngáp
        horizontal_dist = distance(mouth_coordinates[0], mouth_coordinates[3])
        vertical_dist = distance(mouth_coordinates[1], mouth_coordinates[4])
        yawning_ratio = vertical_dist / horizontal_dist
        return yawning_ratio
    return 0.0

def detect_light_condition_and_adjust(frame):
    # Chuyển đổi sang không gian màu LAB để phân tích L-channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Phân tích histogram của L-channel
    l_hist = cv2.calcHist([l], [0], None, [256], [0, 256])
    if np.mean(l_hist[:128]) > np.mean(l_hist[128:]):
        # Điều kiện ánh sáng yếu, cần tăng sáng
        l = cv2.equalizeHist(l)
    adjusted_frame = cv2.merge([l, a, b])
    adjusted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_LAB2BGR)
    return adjusted_frame

def reduce_noise_with_blur(frame):
    # Sử dụng Gaussian blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

class VideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.facemesh_model = get_mediapipe_app()
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": self.GREEN,
            "play_alarm": False,
        }
        self.EAR_txt_pos = (10, 30)
        self.eye_blinks = 0
        self.eye_opens = 0
        self.eye_open_time = time.perf_counter()

    def process(self, frame: np.array, thresholds: dict):
        frame = adjust_image(frame)
        frame = detect_light_condition_and_adjust(frame)
        frame = reduce_noise_with_blur(frame)
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape
        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])

            if EAR < thresholds["EAR_THRESH"]:
                self.eye_blinks += 1
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
                    pygame.mixer.Sound.play(alarm_sound)
            else:
                self.eye_opens += 1
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            if time.perf_counter() - self.eye_open_time > 30:
                if self.eye_opens + self.eye_blinks > 0:
                    open_ratio = (self.eye_opens / (self.eye_opens + self.eye_blinks)) * 100
                else:
                    open_ratio = 0
                open_ratio_txt = f"Eye Open Ratio: {open_ratio:.2f}%"
                plot_text(frame, open_ratio_txt, (10, 60), self.state_tracker["COLOR"])

                if open_ratio < 80:
                    fatigue_warning = "Driver may need to rest and pay attention to health."
                    plot_text(frame, fatigue_warning, (10, 90), self.RED)

                self.eye_opens, self.eye_blinks = 0, 0
                self.eye_open_time = time.perf_counter()

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])
        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]

def run_on_webcam():
    video_capture = cv2.VideoCapture(0)
    video_handler = VideoFrameHandler()
    thresholds = {"EAR_THRESH": 0.28, "WAIT_TIME": 2.0}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        processed_frame, alarm = video_handler.process(frame, thresholds)
        cv2.imshow("Webcam", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def run_on_ip_camera(ip_camera_url):
    video_capture = cv2.VideoCapture(ip_camera_url)
    video_handler = VideoFrameHandler()
    thresholds = {"EAR_THRESH": 0.28, "WAIT_TIME": 2.0}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        processed_frame, alarm = video_handler.process(frame, thresholds)
        cv2.imshow("IP Camera", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Select camera type: (1: Webcam Laptop, 2: Camera IP): ")
    if choice == '1':
        run_on_webcam()
    elif choice == '2':
        ip_url = input("Enter the URL of the IP Camera: ")
        run_on_ip_camera(ip_url)
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    main()
