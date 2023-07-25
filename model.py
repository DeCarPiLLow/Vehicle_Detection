import cv2
import numpy as np
import time
from collections import defaultdict

vid = cv2.VideoCapture("bridge.mp4")

path_label = "coco.names"

classes = []
with open(path_label, 'rt') as f:
    classes = f.read().rstrip('\n').split()

weight_height_target = 320
model_Config = 'yolov3.cfg'
model_weights = 'yolov3.weights'
confThreshold = 0.5
nmsThreshold = 0.3
class_counts = defaultdict(int)

net = cv2.dnn.readNetFromDarknet(model_Config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

output_filename = 'result.mp4'  # Output video filename
fps = 20  # Frames per second for the output video
frame_size = (320, 320)  # Frame size for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

temp_Id = []

def findObject(outputs, img):
    heightTar, weightTar, channelsTar = img.shape
    bbox = []
    classIds = []
    confidences = []
    
    for output in outputs:
        for det in output:
            scores = det[5:14]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            if classId in [2, 3, 5, 7] and confidence > confThreshold:
                w, h = int(det[2] * weightTar), int(det[3] * heightTar)
                x, y = int((det[0] * weightTar) - w / 2), int((det[1] * heightTar) - h / 2)
                xMid, yMid = int((x + (x + w)) / 2), int((y + (y + h)) / 2)
                cv2.circle(img, (xMid, yMid), 1, (0, 0, 255), 1)
                bbox.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(classId)

                if yMid > 162 and yMid < 164 and xMid > 117 and xMid < 176:
                    if classId == 2:
                        class_counts['class1'] += 1
                    elif classId == 3:
                        class_counts['class2'] += 1
                    elif classId == 5:
                        class_counts['class3'] += 1
                    elif classId == 7:
                        class_counts['class4'] += 1

                if yMid > 185 and yMid < 187 and xMid > 180 and xMid < 267:
                    if classId == 2:
                        class_counts['class5'] += 1
                    elif classId == 3:
                        class_counts['class6'] += 1
                    elif classId == 5:
                        class_counts['class7'] += 1
                    elif classId == 7:
                        class_counts['class8'] += 1

    temp_Id = classIds
    draw_box = cv2.dnn.NMSBoxes(bbox, confidences, confThreshold, nmsThreshold)
    if draw_box is not None and len(draw_box) > 0:
        for i in draw_box:
            index = i  # Use 'i' directly as the index
            box = bbox[index]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img, f'{classes[classIds[index]].upper()} {int(confidences[index]*100)}%', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.line(img, (117, 163), (176, 163), (0, 0, 255, 2))
            cv2.line(img, (117, 163), (176, 163), (0, 255, 0, 1))
            cv2.line(img, (117, 163), (176, 163), (0, 255, 0, 1))
            cv2.line(img, (180, 186), (267, 186), (255, 255, 255, 2))
            cv2.line(img, (180, 186), (267, 186), (0, 255, 0, 1))
            cv2.line(img, (180, 186), (267, 186), (0, 255, 0, 1))

    return class_counts

def main():
    starting_time = time.time()

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    video_length_seconds = total_frames // fps

    print("Video Length:", video_length_seconds, "seconds")

    class_counts = defaultdict(int)

    while True:
        ret, img = vid.read()
        if not ret:
            break
        img = cv2.resize(img, (weight_height_target, weight_height_target))
        blob = cv2.dnn.blobFromImage(img, 1/255, (weight_height_target, weight_height_target))
        net.setInput(blob)
        layernames = net.getLayerNames()
        outputnames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputnames)
        class_counts = findObject(outputs, img)  # Remove class_counts from arguments

        result.write(img)

    vid.release()
    result.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    processing_time = end_time - starting_time
    print("Total Time Spent:", processing_time, "seconds")

if __name__ == "__main__":
    main()