import cv2
import mediapipe as mp
import pyautogui
import numpy as np


class EyeTracker:
    def __init__(self, amplification_factor=2.5, smoothing_factor=0.5):
        self.amplification_factor = amplification_factor
        self.smoothing_factor = smoothing_factor
        self.prev_x, self.prev_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,
                                                         min_detection_confidence=0.7,
                                                         min_tracking_confidence=0.7)
        self.screen_w, self.screen_h = pyautogui.size()
        self.eye_landmarks = {
            'left': [469, 470, 471, 472],
            'right': [474, 475, 476, 477]
        }

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame), frame

    def get_eye_position(self, landmarks):
        eye_x, eye_y = 0, 0
        for eye in ['left', 'right']:
            eye_x += np.mean([landmarks[i].x for i in self.eye_landmarks[eye]])
            eye_y += np.mean([landmarks[i].y for i in self.eye_landmarks[eye]])
        return eye_x / 2, eye_y / 2

    def map_to_screen(self, eye_x, eye_y):
        norm_x = (eye_x - 0.5) * 2
        norm_y = (eye_y - 0.5) * 2
        return (
            int(self.screen_w // 2 + norm_x * self.screen_w * (self.amplification_factor / 2)),
            int(self.screen_h // 2 + norm_y * self.screen_h * (self.amplification_factor / 2))
        )

    def apply_smoothing(self, x, y):
        smooth_x = int(self.prev_x * self.smoothing_factor + x * (1 - self.smoothing_factor))
        smooth_y = int(self.prev_y * self.smoothing_factor + y * (1 - self.smoothing_factor))
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    def draw_eye_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        for eye in self.eye_landmarks.values():
            for i in eye:
                lx, ly = int(landmarks[i].x * w), int(landmarks[i].y * h)
                cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

    def track_eyes(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results, frame = self.process_frame(frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    eye_x, eye_y = self.get_eye_position(face_landmarks.landmark)
                    mouse_x, mouse_y = self.map_to_screen(eye_x, eye_y)
                    smooth_x, smooth_y = self.apply_smoothing(mouse_x, mouse_y)
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.1)
                    self.draw_eye_landmarks(frame, face_landmarks.landmark)

            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    EyeTracker().track_eyes()
