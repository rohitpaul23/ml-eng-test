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




def removeText(image):
    # Step 2: OCR to detect and mask the text regions

    #Apply thresholding to increase contrast between text and background
    _, gray_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # Convert to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Dynamic thresholding to enhance contrast
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

    custom_config = r'--oem 3 --psm 11'  # LSTM OCR Engine, Sparse Text
    ocr_result = pytesseract.image_to_data(binary_image, output_type=pytesseract.Output.DICT, config=custom_config)


    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    for i in range(len(ocr_result['text'])):
        (x, y, w, h) = (ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i])
        text = ocr_result['text'][i]
        if len(text.strip()) > 0:  # Mask the text region
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # White out the text region
        
    return image




def preprocessing(image):
    # Step 3: Apply thresholding to make the lines clearer
    _, thresh = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)

    # Step 4: Use morphological operations to remove noise and small irrelevant details
    kernel = np.ones((3, 3), np.uint8)
    #cleaned_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill small gaps
    cleaned_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)  # Remove small noise



    # Using ADAPTIVE_THRESH_GAUSSIAN_C to handle uneven lighting in the image
    binary_image = cv2.adaptiveThreshold(cleaned_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Step 3: Use morphological operations to enhance thicker lines and suppress thinner ones
    # Dilation will emphasize thicker, continuous lines (like walls)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=2)

    # Optionally apply erosion to clean up some noise after dilation
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded





def getWall(eroded):
    # Get the shape of the original image
    height, width = eroded.shape

    # Create a white canvas with the same dimensions
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Step 5: Edge detection (focuses on stronger edges)
    edges = cv2.Canny(eroded, 50, 200, apertureSize=3)

    # Step 6: Hough Line Transform to detect walls
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=100, maxLineGap=5)

    # Step 7: Filter and draw detected wall lines
    for line in lines:
            x1, y1, x2, y2 = line[0]
            line_thickness = np.linalg.norm([x2 - x1, y2 - y1])
            if line_thickness < 10:
                continue  # Ignore measurement lines
            # Draw the detected wall lines
            #print('Drawing')
            cv2.line(white_canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw walls in blue


    result_image = cv2.cvtColor(white_canvas, cv2.COLOR_BGR2GRAY)

    # Step 1: Apply adaptive thresholding or binary thresholding to binarize the image
    _, binary_image = cv2.threshold(result_image, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Apply dilation to merge the parallel lines
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size based on how close the lines are
    dilated = cv2.dilate(binary_image, kernel, iterations=10)

    # Step 3: Apply erosion to thin the lines back to their original width after merging
    eroded = cv2.erode(dilated, kernel, iterations=10)

    return eroded





def overlay(image, wall):
    original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    # Step 2: Apply threshold to extract walls more clearly
    _, binary_walls = cv2.threshold(wall, 127, 255, cv2.THRESH_BINARY)

    # Step 3: Convert the binary image into a color image for overlay
    colored_walls = cv2.cvtColor(binary_walls, cv2.COLOR_GRAY2BGR)

    # Step 4: Create a mask for walls and apply color (e.g., red) to those areas
    mask = binary_walls > 0  # Walls are where the image is not black
    colored_walls[mask] = (0, 0, 255)  # Color the walls red

    # Step 3: Overlay the colored walls onto the original plan
    alpha = 0.6  # Transparency factor for the overlay
    overlay_image = cv2.addWeighted(colored_walls, alpha, original_image, 1 - alpha, 0)

    return overlay_image












def main(pdf_path):
    #pdf_path = r'C:\Users\rohit\Downloads\ml-eng-test\datasets\Walls\A-102 .00 - 2ND  FLOOR PLAN.pdf'

    image = image2pdf(pdf_path)

    imageWOText = removeText(image)

    processedImage = preprocessing(imageWOText)

    wall = getWall(processedImage)

    result = overlay(image, wall)