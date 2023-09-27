# @markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import pandas as pd
import pyautogui
import tkinter as tk
import time
import random

pyautogui.FAILSAFE = False

calibrate = {"Yt": 0, "Yb": 0, "Xl": 0, "Xr": 0, "yc": 0, "xc": 0}


def draw_circle(x, y, radius, line="red"):
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=line, width=4)


# Create the main window
root = tk.Tk()
root.title("Circle on Screen")

root.attributes("-fullscreen", True)

# Create a canvas to draw on
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()


# Specify the coordinates and radius for the circle
# circle_x = root.winfo_screenwidth() // 2  # Center X
# circle_y = root.winfo_screenheight() // 2  # Center Y
# circle_radius = min(root.winfo_screenwidth(), root.winfo_screenheight()) // 4  # Radius is a quarter of the screen size

# Draw the circle on the canvas
# draw_circle(root.winfo_screenwidth() // 2, 10, 10)


def exit_fullscreen(event):
    root.attributes("-fullscreen", False)
    root.quit()


root.bind("<Escape>", exit_fullscreen)


class AVG:
    # n = 10
    # i = 0
    # l = []

    def __init__(self):
        self.n = 10
        self.i = 0
        self.l = []

    def calc(self, num):
        if len(self.l) < self.n:
            self.l.append(num)
        else:
            self.l[self.i] = num
            self.i += 1
            if self.i == self.n:
                self.i = 0
        return sum(self.l) / self.n


ch = 0
c = 0

randx = 0
randy = 0
randc = 0

ddif = AVG()
nif = AVG()

xdifs = AVG()
ydifs = AVG()


cs = list(range(0, (478 * 3) + 2))
# print(len(cs))
df = pd.DataFrame([], columns=cs)


def saveData(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    ss = pyautogui.size()
    x = []

    if len(face_landmarks_list) > 0:
        loc = pyautogui.position()
        ir = face_landmarks_list[0]
        for ri in ir:
            x.append(ri.x)
            x.append(ri.y)
            x.append(ri.z)
        x.append(loc.x / ss.width)
        x.append(loc.y / ss.height)
        df.loc[len(df), :] = x

    return annotated_image


def draw_iris(rgb_image, detection_result, ch=0, c=0):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    s = annotated_image.shape

    ch = ch + 1

    # c = 143

    if len(face_landmarks_list) > 0:
        # print(len(face_landmarks_list[0]))
        ss = pyautogui.size()
        # Main 151
        head = face_landmarks_list[0][168]
        right_eye_c = face_landmarks_list[0][133]
        right_eye_uplid = face_landmarks_list[0][145:146]
        right_eye_downlid = face_landmarks_list[0][159:160]
        right_eye_ball = face_landmarks_list[0][468:469]

        left_eye_c = face_landmarks_list[0][362]
        left_eye_uplid = face_landmarks_list[0][386:387]
        left_eye_downlid = face_landmarks_list[0][374:375]
        left_eye_ball = face_landmarks_list[0][151:152]

        nose_center = face_landmarks_list[0][1]
        nose_top = face_landmarks_list[0][168]

        n1 = math.dist([nose_top.x, nose_top.y], [nose_center.x, nose_center.y])
        nans = nif.calc(n1 * 100)
        an = round(nans * 10) - 80
        # print(round(nans*10))
        # ndif = round((n1)/3)*3
        # ndif -= 80

        d1 = math.dist([head.x, head.y], [right_eye_c.x, right_eye_c.y]) * 100.00
        d2 = math.dist([head.x, head.y], [left_eye_c.x, left_eye_c.y]) * 100.00
        dif = (d1 - d2) * 100
        answer = ddif.calc(dif)
        answer = round(answer / 5) * 5
        # print()
        pyautogui.moveTo((ss.width * (abs(answer - 250) / 500)) - 10, ss.height * (an / 50))  # * (ndif/50)
        # print(round(dif / 10) * 10)

        # 468:478  eye balls
        ir = face_landmarks_list[0][473:474]
        ir.extend(face_landmarks_list[0][1:2])
        ir.extend(face_landmarks_list[0][168:169])
        for ri in ir:
            x.append(ri.x)
            y.append(ri.y)
            # print(s, ri.x*s[0], ri.y*s[1])
            annotated_image = cv2.circle(annotated_image, (int(ri.x * s[1]), int(ri.y * s[0])), 5, (0, 0, 255),
                                         cv2.FILLED)

        ir = face_landmarks_list[0][c:c + 1]
        # ir.extend(face_landmarks_list[0][133:134])
        # ir.extend(face_landmarks_list[0][362: 363])
        for ri in ir:
            x.append(ri.x)
            y.append(ri.y)
            # print(s, ri.x*s[0], ri.y*s[1])
            annotated_image = cv2.circle(annotated_image, (int(ri.x * s[1]), int(ri.y * s[0])), 5, (255, 0, 0),
                                         cv2.FILLED)

    if ch % 50 == 0:
        c += 1
        # print(c)
    return annotated_image, ch, c


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


# STEP 1: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

x = []
y = []

vid = cv2.VideoCapture(0)

while (True):
    break
    ret, frame = vid.read()
    # define the alpha and beta
    alpha = 1.5  # Contrast control
    beta = 10  # Brightness control

    # call convertScaleAbs function
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    f = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(f)
    # face_landmarks_list = detection_result.face_landmarks
    annotated_image2, ch, c = draw_iris(f.numpy_view(), detection_result, ch, c)
    # annotated_image2 = saveData(f.numpy_view(), detection_result)
    annotated_image = draw_landmarks_on_image(f.numpy_view(), detection_result)
    cv2.imshow('frame', annotated_image2)
    randc += 1
    if randc % 100 == 0:
        pass
        # ss = pyautogui.size()
        # pyautogui.moveTo(random.random()*ss.width, random.random()*ss.height)
        # time.sleep(3)
        # break

    # print(pyautogui.size())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# After the loop release the cap object
# df.to_csv("dataset3.csv")

def track():
    print("GOO")
    # return 0
    ss = pyautogui.size()
    while (True):
        xv, yv = getCords()
        xv = xdifs.calc(xv)
        yv = ydifs.calc(yv)

        if xv>calibrate['xc']:
            xv = (xv-calibrate['xc'])/(calibrate['Xl']-calibrate["xc"])
            xv = (ss.width/2) - (xv*(ss.width/2))
        else:
            xv = (calibrate['xc']-xv)/(calibrate['xc']-calibrate['Xr'])
            xv = (xv * (ss.width / 2)) + (ss.width/2)

        if yv>calibrate['yc']:
            yv = (yv-calibrate['yc'])/(calibrate['Yb']-calibrate['yc'])
            yv = (yv*(ss.height/2)) + (ss.height/2)
        else:
            yv = (calibrate['yc']-yv)/(calibrate['yc']-calibrate['Yt'])
            yv = (ss.height/2) - (yv*(ss.height/2))

        pyautogui.moveTo(xv, yv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def getY(face_landmarks_list):
    nose_center = face_landmarks_list[0][1]
    nose_top = face_landmarks_list[0][168]

    n1 = math.dist([nose_top.x, nose_top.y], [nose_center.x, nose_center.y])
    nans = nif.calc(n1 * 100)
    an = round(nans * 10)
    return an


def getX(face_landmarks_list):
    head = face_landmarks_list[0][168]
    right_eye_c = face_landmarks_list[0][133]
    left_eye_c = face_landmarks_list[0][362]
    d1 = math.dist([head.x, head.y], [right_eye_c.x, right_eye_c.y]) * 100.00
    d2 = math.dist([head.x, head.y], [left_eye_c.x, left_eye_c.y]) * 100.00
    dif = (d1 - d2) * 100
    return dif


def getCords():
    ret, frame = vid.read()
    f = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(f)
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) > 0:
        Yp = getY(face_landmarks_list)
        Xp = getX(face_landmarks_list)

        return Xp, Yp
    return 0, 0
    # annotated_image2, ch, c = draw_iris(f.numpy_view(), detection_result)


def calibrate1():
    w = root.winfo_screenwidth()  # Center X
    h = root.winfo_screenheight()  # Center Y
    # calibrate = {"Yt": 0, "Yb": 0, "Xl": 0, "Xr": 0}
    global calibrate

    cali = 0

    # Mid Both
    draw_circle(w // 2, h // 2, 20)
    root.update()
    time.sleep(2)
    draw_circle(w // 2, h // 2, 20, "black")
    root.update()
    while (cali < 100):
        xp, yp = getCords()
        calibrate["Yt"] = yp
        calibrate["Yb"] = yp
        calibrate["Xl"] = xp
        calibrate['Xr'] = xp
        calibrate["yc"] = yp
        calibrate['xc'] = xp
        cali += 1
    cali = 0

    # Top Y
    draw_circle(w // 2, 20, 20)
    root.update()
    time.sleep(2)
    draw_circle(w // 2, 20, 20, "black")
    root.update()
    while (cali < 100):
        xp, yp = getCords()
        if yp < calibrate["Yt"]:
            calibrate["Yt"] = yp
        cali += 1
    cali = 0

    # Y Bottom
    draw_circle(w // 2, h - 20, 20)
    root.update()
    time.sleep(2)
    draw_circle(w // 2, h - 20, 20, "black")
    root.update()
    while cali < 100:
        xp, yp = getCords()
        if yp > calibrate["Yb"]:
            calibrate["Yb"] = yp
        cali += 1
    cali = 0

    # X left
    draw_circle(20, h // 2, 20)
    root.update()
    time.sleep(2)
    draw_circle(20, h // 2, 20, "black")
    root.update()
    while (cali < 100):
        xp, yp = getCords()
        if xp > calibrate["Xl"]:
            calibrate["Xl"] = xp
        cali += 1
    cali = 0

    # X right
    draw_circle(w - 20, h // 2, 20)
    root.update()
    time.sleep(2)
    draw_circle(w - 20, h // 2, 20, "black")
    root.update()
    while (cali < 100):
        xp, yp = getCords()
        if xp < calibrate["Xr"]:
            calibrate["Xr"] = xp
        cali += 1
    cali = 0

    print(calibrate)

    root.attributes("-fullscreen", False)
    root.quit()
    track()
    vid.release()


root.after(2000, calibrate1)
root.protocol("WM_DELETE_WINDOW", track)
root.mainloop()
