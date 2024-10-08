# Machine Learning Engineer Test: Computer Vision and Object Detection

## Objective
This test aims to assess your skills in computer vision and object detection, with a specific focus on detecting room walls and identifying rooms in architectural blueprints or pre-construction plans.

This test evaluates your practical skills in applying advanced computer vision techniques to a specialized domain and your ability to integrate machine learning models into a simple API server for real-world applications.

Choose one of the visual tasks, one of the text extraction tasks, and the API Server task. We encourage you to submit your tests even if you can’t complete all tasks.

Good luck!


## Full test description
[Senior Machine Learning Engineer.pdf](https://github.com/user-attachments/files/16702909/Senior.Machine.Learning.Engineer.pdf)

## Features

- Upload PDF files containing floor plans or Image.
- Annotate walls or rooms based on user selection.
- Returns annotated images in PNG format.

## Installation

### Prerequisites

- Python 3.6 or higher
- Flask
- OpenCV
- pytesseract
- Other required libraries specified in `requirements.txt`

### Clone the Repository

```bash
git clone https://github.com/your-username/wall-room-annotation-api.git
cd wall-room-annotation-api

## Usage
Start the Flask server:

```bash
python app.py
```

The API will be running at http://localhost:5000.

Use curl or any API client (like Postman) to interact with the API.
## Example cURL
```
curl.exe -X POST -F "file=@test.pdf" "http://localhost:5000/run-inference?type=room" --output annotated_room_image.png
curl.exe -X POST -F "file=@test.pdf" "http://localhost:5000/run-inference?type=wall" --output annotated_room_image.png
curl.exe -X POST -F "file=@test.png" "http://localhost:5000/run-inference?type=room" --output annotated_room_image.png
```



