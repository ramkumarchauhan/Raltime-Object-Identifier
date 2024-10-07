# Real-Time Object Identifier

## Overview

The Real-Time Object Identifier is a web application that utilizes a webcam to perform real-time object detection using the YOLOv5 model. Detected objects are visualized in a live video feed, and detailed reports are generated and stored in a MongoDB database.

### Features

- Real-time video feed with object detection.
- Capture and download detection results in text and PDF formats.
- Store detection results in MongoDB with timestamps.
- View historical detection results through a web interface.
- Live updates of detected object counts displayed in a bar chart.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- Flask
- PyTorch
- YOLOv5 model
- MongoDB
- OpenCV
- FPDF (for PDF generation)

## Installation

1. **Clone the Repository**

   ```
   git clone https://github.com/ultralytics/yolov5.git
   cd realtime_object_identifier
   ```

2. **Set Up a Virtual Environment**

   ```
   python -m venv venv
   On Windows use: .\venv\Scripts\activate
   ```

3. **Install Required Packages**

   Create a `requirements.txt` file in the root directory with the following content:

   ```
   Flask
   torch
   torchvision
   pymongo
   opencv-python
   fpdf
   ```

   Then, run:

   ```
   pip install -r requirements.txt
   ```

4. **Start MongoDB**

   Make sure your MongoDB server is running. If you're using a local installation, you can start it with:

   ```
   mongod
   ```

5. **Run the Application**

   ```
   python app.py
   ```

6. **Access the Web Application**

   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

- The webcam feed will be displayed on the main page, showing detected objects in real-time.
- Use the provided buttons to download the detection results or view stored results.
- The application stores detection data in MongoDB, which can be accessed via the `/stored_results` route.

## Directory Structure

```
realtime_object_identifier/
├── app.py
├── requirements.txt
└── templates/
    ├── index.html
    └── stored_results.html
```

## Future Enhancements

1. **Enhanced Visualization**:
   - Integrate more advanced data visualization libraries (e.g., Plotly) for interactive charts.

2. **Model Training**:
   - Add functionality to train the YOLOv5 model with custom datasets to improve accuracy for specific applications.

3. **User Authentication**:
   - Implement user authentication to manage access to stored results and download options.

4. **Performance Optimization**:
   - Optimize the video processing pipeline to improve frame rates and reduce latency.

5. **Deployment**:
   - Create Docker support for easier deployment in various environments.
   - Consider deploying the application on cloud platforms (e.g., AWS, Heroku).

6. **API Integration**:
   - Build RESTful APIs for integration with other services or applications.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize any sections further to better fit your application's specifics or to add any additional features you may have in mind! If you need further assistance, let me know!