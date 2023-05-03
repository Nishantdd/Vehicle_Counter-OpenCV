# Vehicle_Counter-OpenCV
This is a Python script that uses OpenCV to detect vehicles in a video stream. The script reads in a video file and applies a background subtraction algorithm to extract moving objects from the video. It then uses morphological operations to further refine the object detection and identify vehicles in the video stream. The script draws bounding boxes around the detected vehicles and displays a counter of the number of vehicles that have crossed a specified line in the video stream.

The script also includes functionality to handle the case where multiple vehicles are detected at once. It stores the center coordinates of each detected vehicle and compares them against a count line to determine if a vehicle has crossed the line. If a vehicle is detected crossing the line, the script increments a counter and updates the display accordingly.

This script can be used for a variety of applications, including traffic monitoring, surveillance, and security systems that require vehicle detection. It requires Python 3.x and OpenCV installed on your system to run. You can modify the script to work with different video streams and adjust the parameters for vehicle detection to suit your specific needs.
