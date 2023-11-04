import cv2
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Define the scale factors
scales = [1.1, 1.3, 1.5, 1.7]

# Load the cascade classifier for face detection
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    detector = MTCNN()

    # cv2.imshow('grayscale image', gray)
    #cv2.waitKey(0)

    # Detect faces in the frame
    faces = detector.detect_faces(frame)


    # Loop through the detected faces
    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y + h, x:x + w]

        # Resize the face image to (100, 100)
        face_img = cv2.resize(face_img, (100, 100))

        # Normalize the pixel values
        face_img = face_img / 255.0

        # Reshape the image to have an additional channel dimension
        face_img = face_img.reshape((1, 100, 100, 3))

        # Use the model to make a prediction
        mask_prediction = model.predict(face_img)

        # Draw a box around the face
        if mask_prediction >= 0.5:
            color = (0, 255, 0) # Green
            text = "mask"+str(mask_prediction)
        else:
            color = (0, 0, 255) # Red
            text = "no mask"+str(1-mask_prediction)
        # color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Put text on the frame
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 2)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord('q'):
        break


# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
