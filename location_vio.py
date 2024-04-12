import requests
from keras.models import load_model
from collections import deque
import numpy as np
import cv2
from datetime import datetime
import pytz
from PIL import Image
from PIL import ImageEnhance
import telepot
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN


# Function to get current time
def getTime():
    IST = pytz.timezone('Asia/Kolkata')
    timeNow = datetime.now(IST)
    return timeNow


# Function to enhance image
def imgenhance():
    image1 = Image.open('savedImage.jpg')
    curr_bri = ImageEnhance.Sharpness(image1)
    new_bri = 1.3
    img_brightened = curr_bri.enhance(new_bri)
    im1 = img_brightened.save("bright.jpg")

    image2 = Image.open('bright.jpg')
    curr_col = ImageEnhance.Color(image2)
    new_col = 1.5
    img_col = curr_col.enhance(new_col)
    im2 = img_col.save("finalImage.jpg")


# Function to draw faces on image
def draw_faces(filename, result_list):
    data = pyplot.imread(filename)
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        pyplot.subplot(1, len(result_list), i + 1)
        pyplot.axis('off')
        pyplot.imshow(data[y1:y2, x1:x2])
    pyplot.savefig("faces.png")
    pyplot.show()


# Function to get location coordinates dynamically based on IP address
def get_location_coordinates():
    try:
        # Fetch IP address
        ip_response = requests.get('https://api.ipify.org?format=json')
        ip_data = ip_response.json()
        ip_address = ip_data['ip']

        # Fetch location coordinates based on IP address
        location_response = requests.get(f'http://ip-api.com/json/{ip_address}')
        location_data = location_response.json()

        if location_data['status'] == 'success':
            return location_data['lat'], location_data['lon']
        else:
            print("Failed to retrieve location coordinates based on IP address.")
            return None, None
    except Exception as e:
        print("Error fetching location coordinates:", e)
        return None, None


# Function to generate map link
def generate_map_link(latitude, longitude):
    return f'https://www.google.com/maps?q={latitude},{longitude}'


# Function to detect violence
def detectViolence():
    trueCount = 0
    imageSaved = 0
    filename = 'savedImage.jpg'
    my_image = 'finalImage.jpg'
    face_image = 'faces.png'
    sendAlert = 0

    print("Loading model ...")
    model = load_model(r'C:\Users\hp\Desktop\vio\modelnew.h5')
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(0)  # Using webcam (change 0 to the index of your webcam if you have multiple)
    writer = None
    (W, H) = (None, None)
    count = 0

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Enhance brightness and color for better face detection
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)

        # Adjust threshold for violence detection
        i = (preds > 0.30)[0]
        label = i

        text_color = (0, 255, 0)  # default : green

        if label:  # Violence prob
            text_color = (0, 0, 255)  # red
            trueCount += 1
        else:
            text_color = (0, 255, 0)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("recordedVideo.avi", fourcc, 30, (W, H), True)

        writer.write(output)
        cv2.imshow("Output", output)

        if trueCount == 25:
            if imageSaved == 0:
                if label:
                    cv2.imwrite(filename, output)
                    imageSaved = 1

            if sendAlert == 0:
                timeMoment = getTime()
                imgenhance()
                pixels = pyplot.imread(my_image)
                detector = MTCNN()
                faces = detector.detect_faces(pixels)
                draw_faces(my_image, faces)

                latitude, longitude = get_location_coordinates()
                if latitude is not None and longitude is not None:
                    bot = telepot.Bot('your bot id')
                    telegram_group_id = "your group id"
                    map_link = generate_map_link(latitude, longitude)
                    message = f"VIOLENCE ALERT!! \nLATITUDE: {latitude} \nLONGITUDE: {longitude} \nTIME: {timeMoment}\nMap Link: {map_link}"
                    bot.sendMessage(messageid, message)
                    bot.sendPhoto(messageid, photo=open('finalImage.jpg', 'rb'))
                    bot.sendMessage(messageid, "FACES OBTAINED")
                    bot.sendPhoto(messageid, photo=open('faces.png', 'rb'))

                    sendAlert = 1
                else:
                    print("Failed to retrieve location coordinates based on IP address.")

        # Reset alerting variables
        if trueCount >= 25:
            trueCount = 0
            imageSaved = 0
            sendAlert = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


# Call detectViolence function to start real-time violence detection using webcam
detectViolence()
