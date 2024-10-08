import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def image2pdf(pdf_path):
    
    images = convert_from_path(pdf_path)

    # Process each extracted image
    for page_number, img in enumerate(images):
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
    return image


# Function to detect text and draw bounding boxes for rooms
def detect_rooms(plan_image, line_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Perform Hough Line Transform to detect walls
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Create an empty image to draw lines on
    line_image = np.zeros_like(plan_image)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw the walls

    
    gray_plan = cv2.cvtColor(plan_image, cv2.COLOR_BGR2GRAY)

    # Detect the text in the plan (to label rooms)
    data = pytesseract.image_to_data(gray_plan, output_type=pytesseract.Output.DICT)
    
    # Define the room labels we are interested in
    room_labels = ["bedroom", "living", "dining", "kitchen"]
    
    room_bboxes = []  # To store the bounding boxes for each room
    
    # Loop through all detected text and their bounding boxes
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i].lower().strip()  # Get the text and convert to lowercase
        if text in room_labels:
            # Get bounding box coordinates for the detected room label
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            
            # Add a room bounding box to the list
            room_bboxes.append((x, y, x + w, y + h, text))
            
            # Draw the text bounding box in the plan image for visualization
            cv2.rectangle(plan_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(plan_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Now, using the text bounding boxes and the lines, predict the room areas
    for bbox in room_bboxes:
        x1, y1, x2, y2, room_name = bbox
        
        # Find lines closest to this bounding box to predict the room boundary
        # (This is a simplification. Can add more complex logic here to group walls)
        extended_bbox = extend_room_bbox(x1, y1, x2, y2, lines, plan_image.shape)
        
        # Color the room
        color_map = {
            "bedroom": (255, 0, 0),
            "living": (0, 255, 0),
            "dining": (0, 0, 255),
            "kitchen": (255, 255, 0)
        }
        cv2.rectangle(plan_image, (extended_bbox[0], extended_bbox[1]), (extended_bbox[2], extended_bbox[3]), color_map[room_name], -1)

    return plan_image

# Extend the bounding box based on nearby lines (this can be fine-tuned)
def extend_room_bbox(x1, y1, x2, y2, lines, image_shape):
    extension = 50  # Arbitrary extension value, can be adjusted
    extended_x1 = max(x1 - extension, 0)
    extended_y1 = max(y1 - extension, 0)
    extended_x2 = min(x2 + extension, image_shape[1])
    extended_y2 = min(y2 + extension, image_shape[0])
    return extended_x1, extended_y1, extended_x2, extended_y2







def main(wall, image):
    #image = image2pdf(pdf_path)
    print('Inside Room')
    #plan_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    #test_line_image_rgb = cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)
    original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Detect rooms and draw bounding boxes
    annotated_image = detect_rooms(original_image, wall)
    
    return annotated_image