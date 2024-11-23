from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QSizePolicy
from PyQt5 import uic , QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import icons.icons_rc

def split_digits_with_boxes(image_path, digit_size=(28, 28)):
    """
    Splits an image into individual digit images using OpenCV and returns
    both the digit images and their bounding boxes.

    Args:
        image_path (str): Path to the image file.
        digit_size (tuple): Desired size for each digit image (default: (28, 28)).

    Returns:
        tuple: (List of digit images, list of bounding boxes)
               Bounding boxes are in the format (x, y, w, h).
    """
    # Load the image in grayscale and as color for displaying boxes later
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_images = []
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
        bounding_boxes.append((x, y, w, h))     # Save bounding box for drawing

        # Extract the digit and resize it to the model's input size
        digit_img = img_gray[y:y + h, x:x + w]
        digit_img = cv2.resize(digit_img, digit_size)
        digit_images.append(digit_img)

    return img_color, digit_images, bounding_boxes  # Return the color image for display

class Predict(QMainWindow):
    def __init__(self):
        super(Predict, self).__init__()
        uic.loadUi('./predictNumUI.ui', self)

        # remove windows title bar
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        # set main background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # button clicks on top bar
        # minimize window
        self.btnMinus.clicked.connect(self.showMinimized)
        # Close window
        self.btnClose.clicked.connect(self.close)



        # upload btn func
        self.btnUpload.clicked.connect(self.upload)
        self.btnPredict.clicked.connect(self.predict)

        self.imgPath = ''

    def upload(self):
        # Create custom QFileDialog
        dialog = QFileDialog()

        # Set filter for PNG and JPG files
        dialog.setNameFilters(["PNG Files (*.png);;JPG Files (*.jpg *.jpeg);;"])

        # Show dialog and get selected file
        if dialog.exec_():
            file_path = dialog.selectedFiles()[0]

            # Open and display the image
            pixmap = QPixmap(file_path)
            self.imgPath = file_path
            label = self.lblImage
            label.setPixmap(pixmap)

            # Set scaled contents and size policy
            label.setScaledContents(True)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            # Scale pixmap to fit within label dimensions
            width = label.width()
            height = label.height()
            scaled_pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)


    def predict(self):
        try:

            image_path = str(self.imgPath)

            # Load a model
            model = keras.models.load_model('./sudoscan.h5')  # Load your model

            original_img, digit_images, bounding_boxes = split_digits_with_boxes(image_path)
            predicted_digits = []
            for i, digit_img in enumerate(digit_images):
                digit_img = digit_img / 255.0  # Normalize
                digit_img = digit_img.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)

                # Make prediction
                prediction = model.predict(digit_img)
                predicted_digit = np.argmax(prediction)
                predicted_digits.append(predicted_digit)

                # Draw bounding box and prediction text on the original image
                x, y, w, h = bounding_boxes[i]
                cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box around the digit
                cv2.putText(original_img, str(predicted_digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                            2)  # Blue text above digit

            output_path = "./result/annotated_image.jpg"
            cv2.imwrite(output_path, original_img)
            print(f"Annotated image saved at: {output_path}")

            pixmap = QPixmap(output_path)
            label = self.lblIresult
            label.setPixmap(pixmap)
            # Set scaled contents and size policy
            label.setScaledContents(True)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            # Scale pixmap to fit within label dimensions
            width = label.width()
            height = label.height()
            scaled_pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(e)

        # main window drag funcs

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.offset = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.offset is not None and event.buttons() == QtCore.Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.offset = None
        super().mouseReleaseEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = Predict()
    UIWindow.show()
    app.exec_()