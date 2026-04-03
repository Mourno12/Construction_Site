import cv2
from tracking_detector import TrackingDetector

def process_webcam():

    detector = TrackingDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        frame = detector.draw_results(frame, results)

        cv2.imshow("PPE Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return {"success": True}