import cv2
import mediapipe as mp
import math
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        
        self.mode = mode 
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

        self.pushup_count = 0  # Initialize push-up count
        self.bicep_curl_count = 0  # Initialize bicep curl count
        self.bicep_curl_direction = 0  # Initialize direction for curls

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
        
    def findAngle(self, img, p1, p2, p3, draw=True):   
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]
        
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        elif angle > 180:
            angle = 360 - angle
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def detectPushUp(self, img):
        # Check if the necessary landmarks are available
        if len(self.lmList) > 0:
            left_shoulder = 11
            left_elbow = 13
            left_wrist = 15
            right_shoulder = 12
            right_elbow = 14
            right_wrist = 16
            
            # Find angles
            left_elbow_angle = self.findAngle(img, left_shoulder, left_elbow, left_wrist, draw=False)
            right_elbow_angle = self.findAngle(img, right_shoulder, right_elbow, right_wrist, draw=False)
            
            # Determine push-up position
            if left_elbow_angle < 160 and right_elbow_angle < 160:
                pushup_status = "true"
                self.direction = 0  # Reset direction
            elif left_elbow_angle > 160 and right_elbow_angle > 160:
                pushup_status = "true"
                if self.direction == 0:  # Count full push-up
                    self.pushup_count += 1
                    self.direction = 1  # Change direction
            else:
                pushup_status = "false"

            # Show the number of push-ups
            cv2.putText(img, f'Push-up: {pushup_status}', (10, 100), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    def detectSquat(self, img):
        # Angle references for squat detection
        angle_ref = {
            'L': {'hip': [11, 23, 25], 'knee': [23, 25, 27]},
            'R': {'hip': [12, 24, 26], 'knee': [24, 26, 28]}
        }

        # Check if landmarks are available
        if len(self.lmList) > 0:
            # Calculate angles for left leg
            left_knee_angle = self.findAngle(img, *angle_ref['L']['knee'])

            # Calculate angles for right leg
            right_knee_angle = self.findAngle(img, *angle_ref['R']['knee'])

            # Determine squat position
            if left_knee_angle < 90 and right_knee_angle < 90:
                squat_status = "true"
            elif left_knee_angle > 90 and right_knee_angle > 90:
                squat_status = "true"
            else:
                squat_status = "false"

            cv2.putText(img, f'Squat: {squat_status}', (10, 150), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    def detectBicepCurl(self, img, arm='right'):
        # Check if landmarks are available
        if len(self.lmList) > 0:
            if arm == 'right':
                shoulder = 12
                elbow = 14
                wrist = 16
            else:  # left
                shoulder = 11
                elbow = 13
                wrist = 15

            # Find the angle for the chosen arm
            angle = self.findAngle(img, shoulder, elbow, wrist, draw=False)

            # Calculate percentage of curl
            percentage = np.interp(angle, (210, 310), (0, 100))

            # Check for bicep curl count
            if percentage == 100:
                if self.bicep_curl_direction == 0:  # Curl Up
                    self.bicep_curl_count += 1  # Count full curl
                    self.bicep_curl_direction = 1  # Change direction
            if percentage == 0:
                if self.bicep_curl_direction == 1:  # Curl Down
                    self.bicep_curl_direction = 0  # Change direction

            curl_status = "false"
            if self.bicep_curl_direction == 0:  # Curl Down
                curl_status = "true"
            elif self.bicep_curl_direction == 1:  # Curl Up
                curl_status = "true"

            # Show the number of curls
            cv2.putText(img, f'Curl: {curl_status}', (10, 200), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:    
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            detector.detectPushUp(img)
            detector.detectSquat(img)
            detector.detectBicepCurl(img, arm='right')  # Change to 'left' for left arm curls
            cv2.imshow('Pose Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
