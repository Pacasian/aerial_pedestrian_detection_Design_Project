from flask import Flask, render_template,make_response, request, redirect, url_for
import numpy as np
import argparse
import time
import cv2
import os
from flask_autoindex import AutoIndex



# show the output image
#cv2.imshow("Image", image)
#cv2.waitKey(0)
sumith=[]





app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/button1', methods=["POST"])
def show():
	destdir = "static/img/"
	files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ]
	print(files)
	return render_template("index.html",sumith1=files)

@app.route('/button', methods=["POST"])
def gettingText():
    sumith=compute()
    return render_template("index.html", sumith=sumith)
def compute():
	yolo="static/yolo-coco"
	imag="static/img/1.jpg"
	labelsPath = os.path.sep.join([yolo, "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
	weightsPath = os.path.sep.join([yolo, "yolov3.weights"])
	configPath = os.path.sep.join([yolo, "yolov3.cfg"])
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	image = cv2.imread(imag)
	(H, W) = image.shape[:2]
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
	    for detection in output:
	        scores = detection[5:]
	        classID = np.argmax(scores)
	        confidence = scores[classID]

			
	        if confidence > 0.5:
	            box = detection[0:4] * np.array([W, H, W, H])
	            (centerX, centerY, width, height) = box.astype("int")
	            x = int(centerX - (width / 2))
	            y = int(centerY - (height / 2))

	            boxes.append([x, y, int(width), int(height)])
	            confidences.append(float(confidence))
	            classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)

	if len(idxs) > 0:
	    for i in idxs.flatten():
	        (x, y) = (boxes[i][0], boxes[i][1])
	        (w, h) = (boxes[i][2], boxes[i][3])
	        color = [int(c) for c in COLORS[classIDs[i]]]
	        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
	        frame=text
	        if (text.split(":")[0])=="person":
	            #rop_img = image[y:y+h, x:x+w]
	            #frame=boun(crop_img)
	            sumith.append(frame)
	        cv2.putText(image, frame, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	        cv2.imwrite('static/newImage.png',image)
	return (sumith)        

if __name__ == '__main__':
    #app.secret_key='secret123'
    app.run(debug=True)
