import cv2
import mediapipe as mp
import math

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

    def findAngle(self, p1, p2, p3):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]
        
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        return angle

    def printAnglesAndPositions(self):
        if self.results.pose_landmarks:
            # Print positions of all landmarks
            print("Landmark Positions:")
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                print(f'Landmark {id}: (x: {lm.x:.2f}, y: {lm.y:.2f}, z: {lm.z:.2f})')
            
            # Calculate and print angles between certain points (example pairs)
            # Example: Shoulders, Elbows, and Wrists
            left_shoulder = 11
            left_elbow = 13
            left_wrist = 15
            right_shoulder = 12
            right_elbow = 14
            right_wrist = 16
            
            left_elbow_angle = self.findAngle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self.findAngle(right_shoulder, right_elbow, right_wrist)
            
            print(f'Left Elbow Angle: {left_elbow_angle:.2f}')
            print(f'Right Elbow Angle: {right_elbow_angle:.2f}')

def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:    
            img = detector.findPose(img)
            detector.findPosition(img)
            detector.printAnglesAndPositions()
            cv2.imshow('Pose Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
