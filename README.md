# Face Mask Detector
AI model of Mask Recognition Project conducted during the "Deep Learning" class offered by Télecom SudParis. 
The final model contains a real-time face mask detector using Tensorflow, Keras, and OpenCV with your webcam.

## Dataset
The dataset used is provided by Prajna Bhandary (https://github.com/prajnasb/). It contains 1376 images divided into with mask and without mask

## Image Classifier
- Built using the VGG16 CNN Architecture with layers added on top of the base model through transfer learning.
  
### Results
The following results were obtained using our model for artificially generated masks and real masks.
-![Screen Shot 2023-11-04 at 6 56 08 PM](https://github.com/thaisstein/mask-detector/assets/52481495/8d833e0c-25bf-4736-9892-ee65a4462578)
-![Screen Shot 2023-11-04 at 6 43 59 PM](https://github.com/thaisstein/mask-detector/assets/52481495/2ece34da-ef1a-497c-9d6e-9320257f66b2)
-![Screen Shot 2023-11-04 at 6 44 05 PM](https://github.com/thaisstein/mask-detector/assets/52481495/a6724c42-3fa8-4fae-95ba-38ddbe777a62)

## Video Classifier
- Built using the MTCNN model for image finding on each frame, a scale-invariant method.

### Results
Below is a link to the video containing a live demonstration of the video results.
-![Screen Shot 2023-11-04 at 6 47 55 PM](https://github.com/thaisstein/mask-detector/assets/52481495/9f92ac99-fde1-46cf-8ba6-597be5d8a996)(https://youtu.be/Z5UPBoSzV4s "Mask DEMO")

