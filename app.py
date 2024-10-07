# from flask import Flask, Response, render_template
# import numpy as np
# import tensorflow as tf
# import cv2

# app = Flask(__name__)

# # Load the pre-trained MobileNetV2 model
# model = tf.keras.applications.MobileNetV2(weights='imagenet')

# # Function to generate frames for video feed
# def gen_frames():
#     camera = cv2.VideoCapture(0)  # Capture from the first camera
#     while True:
#         success, frame = camera.read()  # Read a frame from the camera
#         if not success:
#             break
        
#         # Preprocess the frame for the model
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224))
#         img = np.array(img) / 255.0
#         img = np.expand_dims(img, axis=0)
        
#         # Make predictions
#         predictions = model.predict(img)
#         decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
#         label = decoded_predictions[0][1]  # Get the label of the predicted object

#         # Display the label on the frame
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for video feed
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

#----------------------------------------------------------------------------------------------- 

# from flask import Flask, Response, render_template
# import numpy as np
# import tensorflow as tf
# import cv2

# app = Flask(__name__)

# # Load the pre-trained MobileNetV2 model
# model = tf.keras.applications.MobileNetV2(weights='imagenet')

# # Function to generate frames for video feed
# def gen_frames():
#     camera = cv2.VideoCapture(0)  # Capture from the first camera
#     while True:
#         success, frame = camera.read()  # Read a frame from the camera
#         if not success:
#             break
        
#         # Preprocess the frame for the model
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224))
#         img = np.array(img) / 255.0
#         img = np.expand_dims(img, axis=0)
        
#         # Make predictions
#         predictions = model.predict(img)
#         decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
#         label = decoded_predictions[0][1]  # Get the label of the predicted object
#         confidence = decoded_predictions[0][2]  # Get the confidence score

#         # Format confidence to percentage
#         confidence_percentage = confidence * 100

#         # Display the label and confidence on the frame
#         cv2.putText(frame, f'{label}: {confidence_percentage:.2f}%', (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for video feed
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

# -------------------------------------------------------------------------------------------

# from flask import Flask, Response, render_template
# import numpy as np
# import cv2

# app = Flask(__name__)

# # Load YOLO
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# # Load COCO class labels
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Function to generate frames for video feed
# def gen_frames():
#     camera = cv2.VideoCapture(0)  # Capture from the first camera
#     while True:
#         success, frame = camera.read()  # Read a frame from the camera
#         if not success:
#             break

#         # Prepare the frame for YOLO
#         height, width, _ = frame.shape
#         blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#         net.setInput(blob)
#         outputs = net.forward(output_layers)

#         # Initialize lists for detected boxes, confidences, and class IDs
#         boxes = []
#         confidences = []
#         class_ids = []

#         # Process the outputs
#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]  # Scores for each class
#                 class_id = np.argmax(scores)  # Get the class ID with the highest score
#                 confidence = scores[class_id]  # Confidence for that class
#                 if confidence > 0.5:  # Only consider detections with confidence above threshold
#                     # Get the bounding box coordinates
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         # Apply non-maxima suppression to suppress overlapping boxes
#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         # Draw bounding boxes and labels on the frame
#         for i in range(len(boxes)):
#             if i in indexes:
#                 x, y, w, h = boxes[i]
#                 label = str(classes[class_ids[i]])
#                 confidence = confidences[i]
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for video feed
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)


# -----------------------------------------------------------------------------------

# from flask import Flask, Response, render_template
# import numpy as np
# import cv2
# import torch

# app = Flask(__name__)

# # Load YOLOv5 model (choose the appropriate model, e.g., 'yolov5s' for small)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model

# # Function to generate frames for video feed
# def gen_frames():
#     camera = cv2.VideoCapture(0)  # Capture from the first camera
#     while True:
#         success, frame = camera.read()  # Read a frame from the camera
#         if not success:
#             break

#         # Perform inference on the frame
#         results = model(frame)

#         # Process results and draw bounding boxes
#         for *xyxy, conf, cls in results.xyxy[0]:  # xyxy: (x1, y1, x2, y2), conf: confidence, cls: class id
#             label = f'{model.names[int(cls)]}: {conf:.2f}'
#             x1, y1, x2, y2 = map(int, xyxy)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for video feed
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

# ------------------------------------------------------------------------------------------

# from flask import Flask, Response, render_template, send_file
# import numpy as np
# import cv2
# import torch
# import datetime

# app = Flask(__name__)

# # Load YOLOv5 model (choose the appropriate model)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Global variable to store detected results
# detected_results = []

# def gen_frames():
#     global detected_results
#     camera = cv2.VideoCapture(0)
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Perform inference
#         results = model(frame)

#         # Process results and draw bounding boxes
#         detected_results.clear()  # Clear previous results
#         for *xyxy, conf, cls in results.xyxy[0]:
#             label = f'{model.names[int(cls)]}: {conf:.2f}'
#             detected_results.append(label)  # Store the label and confidence
#             x1, y1, x2, y2 = map(int, xyxy)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Encode the frame
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/download_results')
# def download_results():
#     # Create a text file with detected results
#     filename = f'detected_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
#     with open(filename, 'w') as f:
#         for result in detected_results:
#             f.write(result + '\n')
    
#     return send_file(filename, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)

# ---------------------------------------------------------------------------------------------

# from flask import Flask, Response, render_template, send_file, jsonify
# import numpy as np
# import cv2
# import torch
# import datetime
# from collections import defaultdict
# from fpdf import FPDF

# app = Flask(__name__)

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Global variables to store detected results and counts
# detected_results = []
# detection_counts = defaultdict(int)

# def gen_frames():
#     global detected_results, detection_counts
#     camera = cv2.VideoCapture(0)
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Perform inference
#         results = model(frame)
        
#         detected_results.clear()  # Clear previous results
#         for *xyxy, conf, cls in results.xyxy[0]:
#             label = f'{model.names[int(cls)]}: {conf:.2f}'
#             detected_results.append(label)  # Store the label and confidence
#             detection_counts[model.names[int(cls)]] += 1  # Count detections
            
#             x1, y1, x2, y2 = map(int, xyxy)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Encode the frame
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/download_results')
# def download_results():
#     filename = f'detected_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
#     with open(filename, 'w') as f:
#         for result in detected_results:
#             f.write(result + '\n')
#     return send_file(filename, as_attachment=True)

# @app.route('/detection_counts')
# def detection_counts_route():
#     return jsonify(detection_counts)

# @app.route('/download_report')
# def download_report():
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
    
#     pdf.cell(200, 10, txt="Detection Report", ln=True, align='C')
#     pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

#     pdf.cell(200, 10, txt="Detected Objects:", ln=True)
#     for label, count in detection_counts.items():
#         pdf.cell(200, 10, txt=f"{label}: {count}", ln=True)

#     report_filename = f"detection_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
#     pdf.output(report_filename)

#     return send_file(report_filename, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)

# ---------------------------------------------------------------------------------------------------
#  With MongoDB

from flask import Flask, Response, render_template, send_file, jsonify
import numpy as np
import cv2
import torch
import datetime
from collections import defaultdict
from pymongo import MongoClient
from fpdf import FPDF

app = Flask(__name__)

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')  # Adjust the URI as needed
db = client['object_detection_db']  # Database name
results_collection = db['detection_results']  # Collection name

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Global variables to store detected results and counts
detection_counts = defaultdict(int)

def gen_frames():
    global detection_counts
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform inference
        results = model(frame)
        
        detection_counts.clear()  # Clear previous counts
        detected_objects = []  # List to store detected objects for MongoDB
        
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            detected_objects.append({'label': model.names[int(cls)], 'confidence': conf.item()})
            detection_counts[model.names[int(cls)]] += 1  # Count detections
            
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Store detected objects in MongoDB
        if detected_objects:
            results_collection.insert_one({
                'timestamp': datetime.datetime.now(),
                'detected_objects': detected_objects
            })

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_results')
def download_results():
    filename = f'detected_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(filename, 'w') as f:
        for label, count in detection_counts.items():
            f.write(f"{label}: {count}\n")
    return send_file(filename, as_attachment=True)

@app.route('/detection_counts')
def detection_counts_route():
    return jsonify(detection_counts)

@app.route('/download_report')
def download_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.cell(200, 10, txt="Detected Objects:", ln=True)
    for label, count in detection_counts.items():
        pdf.cell(200, 10, txt=f"{label}: {count}", ln=True)

    report_filename = f"detection_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_filename)

    return send_file(report_filename, as_attachment=True)

@app.route('/stored_results')
def stored_results():
    results = list(results_collection.find())
    return render_template('stored_results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
