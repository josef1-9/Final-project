from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import DeiTFeatureExtractor, DeiTForImageClassification

# Load YOLO model for object detection
model_yolo = YOLO("yolo-Weights/yolov8n.pt")


#  Define the classes for YOLO detection
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "pen",  # Existing classes
    "lamp", "basketball", "soccer ball", "towel", "glasses", "wallet", "headphones", "helmet",
    "guitar", "drums", "keyboard (music)", "microphone", "camera", "binoculars", "television",
    "printer", "scanner", "monitor", "projector", "speaker", "flashlight", "hammer", "screwdriver",
    "pliers", "wrench", "umbrella", "backpack", "briefcase", "wallet", "luggage", "handcuffs",
    "watch", "bracelet", "necklace", "ring", "earrings", "hat", "sunglasses", "mask", "scarf",
    "gloves", "coat", "jacket", "hoodie", "sweater", "dress", "skirt", "pants", "shorts", "shirt",
    "t-shirt", "blouse", "tie", "shoes", "boots", "sandals", "heels", "sneakers", "socks",
    "watermelon", "pineapple", "grapes", "kiwi", "pear", "peach", "plum", "cherry", "strawberry",
    "blueberry", "raspberry", "blackberry", "avocado", "pomegranate", "mango", "papaya", "coconut",
    "fig", "dates", "lemon", "lime", "grapefruit", "cantaloupe", "honeydew", "squash", "cucumber",
    "eggplant", "bell pepper", "jalapeno", "habanero", "onion", "garlic", "ginger", "radish",
    "celery", "asparagus", "artichoke", "spinach", "kale", "arugula", "lettuce", "cabbage",
    "cauliflower", "broccoli", "turnip", "rutabaga", "carrot", "sweet potato", "potato", "yam",
    "beet", "zucchini", "pumpkin", "butternut squash", "acorn squash", "spaghetti squash",
    "sunflower", "rose", "daisy", "tulip", "daffodil", "lily", "orchid", "iris", "dandelion",
    "carnation", "snapdragon", "daisy", "poppy", "aster", "lavender", "peony", "chrysanthemum"

]

# Load the CCT model (DeiT)

num_labels = len(classNames)  # Calculate the number of labels/classes

# Load the CCT model (DeiT) for image classification
cct_model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224', num_labels=num_labels)
# Load the feature extractor for the CCT model
feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
# Set the CCT model to evaluation mode
cct_model.eval()


# Start webcam
cap = cv2.VideoCapture(0)  # Initialize video capture; '0' denotes the default camera or provide specific camera index
cap.set(3, 640)  # Set video width to 640 pixels
cap.set(4, 480)  # Set video height to 480 pixels

# Loop to continuously capture frames from the webcam
while True:
    success, img = cap.read()  # Read a frame from the webcam

    # Perform object detection using YOLO
    results = model_yolo(img, stream=True)  # Detect objects in the frame using YOLO

    for r in results:
        boxes = r.boxes  # Retrieve detected bounding boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract box coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            roi_img = Image.fromarray(img[y1:y2, x1:x2])  # Extract the Region of Interest (ROI) from the frame
            
            # Put a rectangle around the detected object on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Preprocess the Region of Interest for CCT model
            roi_img = feature_extractor(images=roi_img, return_tensors="pt")  # Extract features for the CCT model
            
            # Perform inference with CCT model
            with torch.no_grad():
                outputs = cct_model(**roi_img)  # Get predictions from the CCT model
            
            # Get the predicted label from CCT
            predicted_label = classNames[torch.argmax(outputs.logits).item()]  # Retrieve the predicted label

            # Display YOLO-detected label and CCT-predicted label on the frame
            yolo_label = classNames[int(box.cls[0])]  # Get YOLO-detected label
            cv2.putText(img, f"YOLO: {yolo_label}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img, f"CCT: {predicted_label}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Webcam', img)  # Display the frame with detections and labels
    # Break the loop if the 'Webcam' window is closed
    if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
        break
    key = cv2.waitKey(1)
    if key == 27:  # 'Esc' key code
        break

cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all OpenCV windows







