import cv2
import tensorflow as tf
from mtcnn.mtcnn import MTCNN


# Load the saved model

model = tf.keras.models.load_model('model.h5')

# Load the cascade classifier for face detection
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread('no_mask.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#detector
detector = MTCNN()

# Detect faces in the image
faces = detector.detect_faces(gray)
# Draw a box around each face
for face in faces:
    x, y, w, h = face['box']
    face_img = gray[y:y + h, x:x + w]

    # Resize the face to the input size of the CNN
    face_img = cv2.resize(face_img, (100, 100))

    # Normalize the face image
    face_img = face_img / 255.0

    # Reshape the image to have an additional channel dimension
    face_img = face_img.reshape((1, 100, 100, 3))

    # Use the model to make a prediction
    mask_prediction = model.predict(face_img)

    # Draw a box around the face
    if mask_prediction >= 0.5:
        color = (0, 255, 0) # Green
        text = "mask" + str(mask_prediction[0])
    else:
        color = (0, 0, 255) # Red
        text = "no mask" + str(1-mask_prediction[0])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Put text on the image
    cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

