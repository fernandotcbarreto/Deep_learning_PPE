#https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420
#https://github.com/y3mr3/PPE-Detection-YOLO
#https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208
#https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088

import cv2
import numpy as np 
import argparse
import time

probability_minimum=0.7

def load_yolo():
  #net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
  #net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
  net = cv2.dnn.readNetFromDarknet('yolov3_ppe_test.cfg', 'yolov3_ppe_train_9000.weights')
  #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  classes = []
  with open("class.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
  layers_names = net.getLayerNames()
  output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  return net, classes, colors, output_layers


def load_yolo_person():
  #net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
  #net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
  net = cv2.dnn.readNetFromDarknet('C:/Users/fernando.barreto/Documents/person_detect/yolov3.cfg', 'C:/Users/fernando.barreto/Documents/person_detect/yolov3.weights')
  #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  classes = []
  with open("C:/Users/fernando.barreto/Documents/person_detect/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
  layers_names = net.getLayerNames()
  output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):			
  blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
  net.setInput(blob)
  outputs = net.forward(outputLayers)
  return blob, outputs


def detect_objects_person(img, net, outputLayers):			
  blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
  net.setInput(blob)
  outputs = net.forward(outputLayers)
  return blob, outputs


def get_box_dimensions(outputs, height, width):
  boxes = []
  confs = []
  class_ids = []
  for output in outputs:
    for detect in output:
      scores = detect[5:]
      ##print(scores)
      class_id = np.argmax(scores)
      conf = scores[class_id]
      if conf > probability_minimum:
        center_x = int(detect[0] * width)
        center_y = int(detect[1] * height)
        w = int(detect[2] * width)
        h = int(detect[3] * height)
        x = int(center_x - w/2)
        y = int(center_y - h / 2)
        boxes.append([x, y, w, h])
        confs.append(float(conf))
        class_ids.append(class_id)
  return boxes, confs, class_ids


def get_box_dimensions_person(outputs, height, width):
  boxes = []
  confs = []
  class_ids = []
  for output in outputs:
    for detect in output:
      scores = detect[5:]
      ##print(scores)
      class_id = np.argmax(scores)
      conf = scores[class_id]
      if conf > probability_minimum:
        center_x = int(detect[0] * width)
        center_y = int(detect[1] * height)
        w = int(detect[2] * width)
        h = int(detect[3] * height)
        x = int(center_x - w/2)
        y = int(center_y - h / 2)
        boxes.append([x, y, w, h])
        confs.append(float(conf))
        class_ids.append(class_id)
  return boxes, confs, class_ids



def draw_labels(boxes, confs, colors, class_ids, classes, img, out, x, y): 
  indexes = cv2.dnn.NMSBoxes(boxes, confs, probability_minimum, 0.4)
  font = cv2.FONT_HERSHEY_PLAIN
  x1=x
  y1=y
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      #print(label)
      #type(label)
#      if label=='person':
      #print(label, x,y,w,h)
      #color = colors[i]
      #cv2.rectangle(img, (x+x1,y+y1), (x+w+x1, y+h+y1), color, 2)
      #cv2.putText(img, label, (x+x1, y+y1 - 5), font, 2, color, 2)
      if label.startswith('no'):
        cv2.rectangle(img, (x+x1,y+y1), (x+w+x1, y+h+y1), (0, 0, 255), 2)
        cv2.putText(img, label, (x+x1, y+y1 - 5), font, 2, (0, 0, 255), 2)
      else:
        cv2.rectangle(img, (x+x1,y+y1), (x+w+x1, y+h+y1), (0, 255, 0), 2)
        cv2.putText(img, label, (x+x1, y+y1 - 5), font, 2, (0, 255, 0), 2)

def draw_labels_person(boxes, confs, colors, class_ids, classes, img, out, model2, classes2, colors2, output_layers2): 
  indexes = cv2.dnn.NMSBoxes(boxes, confs, probability_minimum, 0.4)
  font = cv2.FONT_HERSHEY_PLAIN
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      #print(x,y,w,h, len(boxes), len(indexes), len(classes), len(class_ids))
      #print (indexes)
      #print (class_ids)
      xc,xc2,yc,yc2=x, x+w, y, y+h
      #print(1)
      if x<0:
        xc=0
      if x+w>img.shape[1]:
        xc2=img.shape[1]
      if y<0:
        yc=0
      if y+h>img.shape[0]:
        yc2=img.shape[0]        
      #print(2)
      label = str(classes[class_ids[i]])
      #print(yc,yc2, xc, xc2, img.shape)
      frame = img[yc:yc2, xc:xc2]
      #print(img.shape)
      #print(frame.shape)
      #stop
      if label=='person':
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        height2, width2, channels2 = frame.shape
        blob2, outputs2 = detect_objects(frame, model2, output_layers2)
        boxes2, confs2, class_ids2 = get_box_dimensions(outputs2, height2, width2)
        draw_labels(boxes2, confs2, colors2, class_ids2, classes2, img, out, xc, yc)
        if ((x<=180) and  (y<=200)):
          cv2.rectangle(img, (0,0), (200, 200),(0, 0, 255), 2)
          cv2.putText(img, 'DANGER', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
  cv2.imshow("Image", img)
  out.write(img.astype('uint8'))





initime=time.time()

def webcam_detect():
  model, classes, colors, output_layers = load_yolo_person()
  model2, classes2, colors2, output_layers2 = load_yolo()
  initime=time.time()
  timee=time.time() - initime
  while timee<1:
    timee=time.time() - initime
    pass
  initime=time.time()
  cap = cv2.VideoCapture(0)
  out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
  while True:
    _, frame = cap.read()
    timee=time.time() - initime
    if timee>=0:
      height, width, channels = frame.shape
      blob, outputs = detect_objects_person(frame, model, output_layers)
      boxes, confs, class_ids = get_box_dimensions_person(outputs, height, width)
      draw_labels_person(boxes, confs, colors, class_ids, classes, frame, out, model2, classes2, colors2, output_layers2)
      #initime=time.time()
      key = cv2.waitKey(1)
      if key == 27:
        break
      pass
    cv2.imshow("Image", frame)
    out.write(frame.astype('uint8'))
    key = cv2.waitKey(1)
    if key == 27:
      break
    if timee>50:
      break
  cap.release()
  out.release()

webcam_detect()
#cap.release()
#out.release()

cv2.destroyAllWindows()

ffmpeg \
  -i output_PPE.avi \
  -vf scale=512:-1 \
  person_PPE.gif