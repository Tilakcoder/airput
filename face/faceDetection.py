# @markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import pandas as pd
import pyautogui
import time
import random

class AVG:
    n = 50
    i = 0
    l = []

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
c = 150

randx = 0
randy = 0
randc = 0

ddif = AVG()
cs = list(range(0, (478*3)+2))
# print(len(cs))
df = pd.DataFrame([], columns=cs)


def saveData(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    ss = pyautogui.size()
    x=[]

    if len(face_landmarks_list) > 0:
        loc = pyautogui.position()
        ir = face_landmarks_list[0]
        for ri in ir:
            x.append(ri.x)
            x.append(ri.y)
            x.append(ri.z)
        x.append(loc.x/ss.width)
        x.append(loc.y/ss.height)
        df.loc[len(df), :] = x

    return annotated_image


def draw_iris(rgb_image, detection_result, ch, c):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    s = annotated_image.shape

    ch = ch + 1

    # c = 143


    if len(face_landmarks_list) > 0:
        # print(len(face_landmarks_list[0]))
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

        d1 = math.dist([head.x, head.y], [right_eye_c.x, right_eye_c.y]) * 100.00
        d2 = math.dist([head.x, head.y], [left_eye_c.x, left_eye_c.y]) * 100.00
        dif = (d1 - d2) * 100
        answer = ddif.calc(dif)
        print(round(answer/5)*5)
        # print(round(dif / 10) * 10)

        # 468:478  eye balls
        ir = face_landmarks_list[0][473:474]
        # ir.extend(face_landmarks_list[0][159:160])
        # ir.extend(face_landmarks_list[0][386:387])
        # ir.extend(face_landmarks_list[0][374:375])
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

    if ch % 100 == 0:
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


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


# STEP 1: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# image = mp.Image.create_from_file("image.png")

# detection_result = detector.detect(image)

# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# plt.show()
print("ALL set")

x = []
y = []

vid = cv2.VideoCapture(0)

while (True):
    ret, frame = vid.read()
    # define the alpha and beta
    alpha = 1.5  # Contrast control
    beta = 10  # Brightness control

    # call convertScaleAbs function
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    f = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(f)
    # face_landmarks_list = detection_result.face_landmarks
    # annotated_image2, ch, c = draw_iris(f.numpy_view(), detection_result, ch, c)
    annotated_image2 = saveData(f.numpy_view(), detection_result)
    annotated_image = draw_landmarks_on_image(f.numpy_view(), detection_result)
    cv2.imshow('frame', annotated_image2)
    randc += 1
    if randc%100 == 0:
        ss = pyautogui.size()
        pyautogui.moveTo(random.random()*ss.width, random.random()*ss.height)
        time.sleep(3)
        # break

    # print(pyautogui.size())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
df.to_csv("dataset2.csv")
