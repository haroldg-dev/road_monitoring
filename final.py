import argparse
import time
from pathlib import Path

import datetime
import os.path
from os import path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, TracedModel
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()
# class index for our required detection classes
required_class_index = [2, 3, 5, 7]
typeslist = ["auto", "moto", "omnibus", "trailer"]
# Middle cross line position
middle_line_position = 100   
up_line_position = middle_line_position - 30 #70
down_line_position = middle_line_position + 30 #130

# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):
        #print("CARRO ", id, " EN EL AREA BAJANDO,   COORDENADA ", iy)
        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        #print("CARRO ", id, " EN EL AREA SUBIENDO,  COORDENADA ", iy )
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        #print("CARRO ", id, " FUERA DE AREA ARRIBA, COORDENADA ", iy)
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1
            save_data(id, typeslist[index], "subida", 0)

    elif iy > down_line_position:
        #print("CARRO ", id, " FUERA DE AREA ABAJO,  COORDENADA ", iy)
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1
            save_data(id, typeslist[index], "bajada", 0)
    

    # Draw circle in the middle of the rectangle
    #cv2.circle(img, center, 2, colors[index], -1)  # end here
    #print(up_list, down_list)

# Función para almacenar datos
def save_data(id, tipo = "null", direccion = "null", distancia = 0):
    fecha = str(datetime.date.today())
    reporte = "report-" + str(fecha)
    now = datetime.datetime.now()
    t = now.strftime("%H:%M:%S")
    linea = str(id) + ";" + str(t) + ";" + tipo + ";" + direccion + ";" + str(distancia) + "\n"
    # validamos si existe el archivo de reporte del día
    if(path.exists(reporte)):
        # id carro - timestamp - tipo - direccion - distancia
        f = open(reporte, "a")
        f.write(linea)
        f.close()
    else :
        f = open(reporte, "a")
        f.write("ID;FECHA;TIPO;DIRECCION;DISTANCIA\n")
        f.write(linea)
        f.close()

def detect(save_img=False):
    weights = 'yolov7-tiny.pt'
    #source = 'http://root:root@169.254.221.60/mjpg/video.mjpg'
    source = '../../basics/cars.mp4'
    imgsz = 320
    txt_path = './report.txt'
    trace = True
    view_img = True
    save_txt = True
    device = 'cpu'
    conf_thres = 0.60 # porcentaje de acertividad minima
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    save_conf = True
    webcam = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride                  
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()

    print("iniciofor")
    
    for path, img, im0s, vid_cap, img0 in dataset:
        detection = []
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            hh, ww, cc = img0.shape
            hhh = int(hh/3)
            """ print(hhh)
            imgf = img0[0+hhh:hh-hhh,::] """
            
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                """ for c in det[:, -1].unique():
                    if c in required_class_index: 
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string """

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img and cls in required_class_index:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1]) + hhh) , (int(xyxy[2]), int(xyxy[3]) + hhh), colors[int(cls)], 2, lineType=cv2.LINE_AA)
                        detection.append([int(xyxy[0]), int(xyxy[1]), (int(xyxy[2])-int(xyxy[0])), (int(xyxy[3])-int(xyxy[1])), required_class_index.index(int(cls))])
            
                boxes_ids = tracker.update(detection)

                for box_id in boxes_ids:
                    x, y, w, h, id, _= box_id
                    cv2.putText(img0, str(id), (x, y - 15 + hhh), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    count_vehicle(box_id, img0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            
            # Stream results
            if view_img:
                # Draw the crossing lines
                cv2.line(img0, (0, middle_line_position + hhh), (ww, middle_line_position + hhh), (255, 0, 255), 2)
                cv2.line(img0, (0, up_line_position + hhh), (ww, up_line_position + hhh), (0, 0, 255), 2)
                cv2.line(img0, (0, down_line_position + hhh), (ww, down_line_position + hhh), (0, 0, 255), 2)

                # Draw counting texts in the frame
                cv2.putText(img0, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
                cv2.putText(img0, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
                cv2.putText(img0, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
                cv2.putText(img0, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
                cv2.putText(img0, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
                cv2.putText(img0, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
                cv2.imshow(str(p), img0)
                """ while True:
                    cv2.waitKey(1) 
                    a = input()
                    if a == "1":
                        break """
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    
    with torch.no_grad():
        detect()
