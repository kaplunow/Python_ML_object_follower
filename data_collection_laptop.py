import cv2
import os
from uuid import uuid1

root_dir = "dataset_farm"
classes_dirs = {"horse": 'h', "person": 'p', "nothing": 'n'}  # specify names of classes and buttons tied to them
stop_button = 'q'  # button that stops data collection


def create_dirs(root, classes):
    for nn_class in classes:
        try:
            os.makedirs(root + "/" + nn_class)
        except FileExistsError:
            print(f'Directory "{root}/{nn_class}" not created because it already exists')


def classify_frame(button, frame):
    global root_dir
    global classes_dirs
    global stop_button
    if button == ord(stop_button):
        return 0

    for key in classes_dirs:
        if button == ord(classes_dirs[key]):
            cv2.imwrite(f"{root_dir}/{key}/{key + str(uuid1())}.png", frame)
            print_raport(root_dir)
    return 1


def print_raport(root):
    message = ""
    for class_dir in os.listdir(root):
        count = len(os.listdir(f"{root}/{class_dir}"))
        message += f"{class_dir}: {str(count)}   "
    print(message)


create_dirs(root_dir, classes_dirs)

cam = cv2.VideoCapture(0)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)  # set image dimensions

if not cam.isOpened():
    print("ERROR! Cannot open the camera")
    exit()
print("connected successfully")
print_raport(root_dir)
while True:
    ret, frame = cam.read()  # get image form camera
    if not ret:  # check success
        print("ERROR!")
        break

    frame = cv2.resize(frame, (224, 224))
    cv2.imshow("frame", frame)  # show image

    if not classify_frame(cv2.waitKey(1), frame):
        break


cam.release()
cv2.destroyAllWindows()

