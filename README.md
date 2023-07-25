# Vehicle_Detection
As shown in the image above, when the vehicles in the frame are detected, they are counted. After getting detected once, the vehicles get tracked and do not get re-counted by the algorithm.
You may also notice that the vehicles will initially be detected and the counter increments, but for a few frames, the vehicle is not detected, and then it gets detected again. As the vehicles are tracked, the vehicles are not re-counted if they are counted once.

# How To run the program:
Install the yolo-coco v3 source file in the same folder with model.py

wget https://pjreddie.com/media/files/yolov3.weights
