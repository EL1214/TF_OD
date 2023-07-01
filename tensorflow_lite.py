import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Function to perform object detection on images
def detect_objects_image(image, interpreter, input_details, output_details, labels, min_conf=0.5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    detections = []

    # Loop over all detections and filter based on confidence threshold
    for i in range(len(scores)):
        if scores[i] > min_conf:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            class_id = int(classes[i])
            class_name = labels[class_id]
            confidence = int(scores[i] * 100)

            detections.append({"class_name": class_name, "confidence": confidence, "bbox": (xmin, ymin, xmax, ymax)})

    return detections

# Streamlit app code
def main():
    st.title("Object Detection for Self-driving Cars")
    st.write("This app performs object detection using a custom model.")

    # Upload file
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Load the label map into memory
        labels_path = 'PATH_TO_labelmap.pbtxt'    # load the path to labelmap
        labels = []
        with open(labels_path, 'r') as f:
            for line in f.readlines():
                if 'name:' in line:
                    label = line.split(':')[1].strip().strip("'")
                    labels.append(label)

        # Load the TensorFlow Lite model into memory
        model_path = 'PATH_TO_.tflite'      # load the path to model.tflite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if file.type.startswith("image"):
            # Perform object detection on the uploaded image
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
            detections = detect_objects_image(image, interpreter, input_details, output_details, labels)

            # Draw bounding boxes and labels on the image
            for detection in detections:
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                xmin, ymin, xmax, ymax = detection["bbox"]

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                cv2.putText(image, f"{class_name}: {confidence}%", (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2)

            # Display the image with detections
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)        

if __name__ == "__main__":
    main()
st.text("Developed by Weekend Warriors")