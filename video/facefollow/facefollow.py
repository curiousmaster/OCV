#! /usr/bin/python3

# import libraries
import numpy as np
import cv2
import face_recognition

# Get a reference to webcam
video_capture = cv2.VideoCapture("/dev/video0")


# Initialize variables
face_locations = []


def facefollow():
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #rgb_frame = frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            cropimg = frame.copy()

            # ------------------------------
            # Crop image around face
            # ------------------------------
            cropimg = cropimg[top:bottom, left:right]

            # ------------------------------
            # Draw a box around the face
            # ------------------------------
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # ------------------------------
            # Scale output
            # ------------------------------
            height, width = frame.shape[:2]
            dim = (int(height), int(height))

            cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)

            imgconcat = np.concatenate((frame, cropimg), axis=1)

        cv2.imshow('Face', imgconcat)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    facefollow()


if __name__ == "__main__":
    main()
