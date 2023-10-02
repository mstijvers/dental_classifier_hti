import cv2
import dlib
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('./static/shape_predictor_68_face_landmarks.dat')
        self.prev_bbox = None
        self.stable_count = 0
        self.stable_duration = 4
        self.show_cropped_mouth = False  # Flag to control when to show the cropped_mouth.jpg
    def __del__(self):
        self.video.release()

    def get_frame(self):
        if self.show_cropped_mouth:
            # If the cropped_mouth.jpg exists and the flag is True, display it
            frame = cv2.imread('static/images/cropped_mouth.jpg')
        else:
            ret, frame = self.video.read()
            if not ret:
                return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        for face in faces:
            landmarks = self.shape_predictor(gray, face)

            # Extract mouth landmarks
            mouth_landmarks = []
            for i in range(48, 68):  # Mouth landmarks are indexed from 48 to 67 in dlib
                x, y = landmarks.part(i).x, landmarks.part(i).y
                mouth_landmarks.append((x, y))

            # Calculate the bounding box around the mouth
            min_x = min(mouth_landmarks, key=lambda x: x[0])[0]
            max_x = max(mouth_landmarks, key=lambda x: x[0])[0]
            min_y = min(mouth_landmarks, key=lambda x: x[1])[1]
            max_y = max(mouth_landmarks, key=lambda x: x[1])[1]

            if max_x - min_x >= 224 and max_y - min_y >= 224:
                # Draw the bounding box around the mouth
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2)

                # Check if the bounding box is relatively stable for 5 seconds
                if self.prev_bbox is None:
                    self.prev_bbox = (min_x, max_x, min_y, max_y)
                else:
                    if (
                            abs(self.prev_bbox[0] - min_x) < 10 and
                            abs(self.prev_bbox[1] - max_x) < 10 and
                            abs(self.prev_bbox[2] - min_y) < 10 and
                            abs(self.prev_bbox[3] - max_y) < 10
                    ):
                        self.stable_count += 1
                        if self.stable_count >= self.stable_duration:
                            # Save a picture of the stable bounding box
                            cv2.imwrite('mouth_bounding_box.jpg', frame)
                            picture = cv2.imread('mouth_bounding_box.jpg')
                            # Crop the image to the bounding box
                            cropped_teeth = picture[min_y:max_y, min_x:max_x]
                            # Save the cropped image
                            cv2.imwrite('static/images/cropped_mouth.jpg', cropped_teeth)
                            self.toggle_show_cropped_mouth()

                    else:
                        self.stable_count = 0
                    self.prev_bbox = (min_x, max_x, min_y, max_y)
            else:
                # Display a message to "come closer to the webcam"
                cv2.putText(frame, "Come closer to the webcam", (350, 650), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255),
                            2)

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()

    def toggle_show_cropped_mouth(self):
        self.show_cropped_mouth = not self.show_cropped_mouth

    def reset_camera(self):
        # Release the video capture and reinitialize it to reset the camera view
        self.show_cropped_mouth = False
        #self.video.release()
        #self.video = cv2.VideoCapture(0)
