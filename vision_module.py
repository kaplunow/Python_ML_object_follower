import torch
import torchvision
import cv2
import numpy as np
import traitlets
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import Camera, bgr8_to_jpeg
import torch.nn.functional as F
import time

class vision_mod():
    def __init__(self, network_path, out_features_num, main_feature):
        self.model = torchvision.models.alexnet(pretrained=False)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, out_features_num)
        self.model.load_state_dict(torch.load(network_path))
        self.model.eval()
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)  # place model on GPU
        self.camera = Camera.instance(width=224, height=224)
        self.mean = 255.0 * np.array([0.485, 0.456, 0.406])
        self.stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        self.normalize = torchvision.transforms.Normalize(self.mean, self.stdev)
        self.segment_dims = (74, 223)
        self.main_feature = main_feature
    
    
    def deinit(self):
        self.camera.stop()
        
        
    def create_segments(self, frame, dims, overlap):
        width = dims[0]
        height = dims[1]
        rows, columns, channels = frame.shape
        coords_vec = []
        frames_vec = []
        for i in range(0, rows-height, height-overlap):
            for j in range(0, columns-width, width-overlap):
                sub_frame = frame[i:(i+height), j:(j + width)]
                coords_vec.append((i, j))
                frames_vec.append(sub_frame)
        return coords_vec, frames_vec


    def preprocess(self, camera_value):
        x = camera_value
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = self.normalize(x)
        x = x.to(self.device)
        x = x[None, ...]
        return x


    def draw_bounding_boxes(self, frame, draw_list, sub_coords):
        for i in draw_list:
            cv2.rectangle(frame, (sub_coords[i][1], sub_coords[i][0]), (self.segment_dims[0]+sub_coords[i][1], self.segment_dims[1]+sub_coords[i][1]), (0,255,255), 2)            
        return frame


    def get_draw_list(self, frame):
        sub_coords, sub_frames = self.create_segments(frame, self.segment_dims, 37)
        i = 0
        draw_list = []
        for sub_f in sub_frames:
            frame_proc = self.preprocess(cv2.resize(sub_f, (224, 224)))
            y = self.model(frame_proc)
            y = F.softmax(y, dim=1)
            prob = float(y.flatten()[self.main_feature])
            if prob > 0.7:
                draw_list.append(i)
                break
            i += 1
        return draw_list, sub_coords


    def get_pos_with_feed(self):
        frame = self.camera.value
        draw_list, sub_coords = self.get_draw_list(frame)
        self.draw_bounding_boxes(frame, draw_list, sub_coords)
        time.sleep(0.001)
        return draw_list, frame


    def get_pos(self):
        frame = self.camera.value
        draw_list, sub_coords = self.get_draw_list(frame)
        time.sleep(0.001)
        return draw_list
