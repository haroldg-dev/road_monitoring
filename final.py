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
from utils.torch_utils import time_synchronized, TracedModel
from tracker import *

# Iniciamos el tracker
tracker = EuclideanDistTracker()
# Clases requeridas para el reconocimiento
required_class_index = [2, 5, 7]
typeslist = ["vehículo liviano", "Vehículo mediano", "Vehículo pesado"]
# Lineas para el análisis de velocidad
middle_line_position = 100   
up_line_position = middle_line_position - 30 #70
down_line_position = middle_line_position + 30 #130

# Listas para almacenar el conteo de vehículos
time_cero = []
time_final = []
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0]
down_list = [0, 0, 0]

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Función para almacenar datos
def save_data(id, tipo = "null", velocidad = 0):
    fecha = str(datetime.date.today())
    reporte = "/home/vision/reportes/reporteVehiculos.csv"
    now = datetime.datetime.now()
    t = now.strftime("%H:%M:%S")
    linea = str(fecha) + ";" + str(t) + ";" + tipo + ";" + str(velocidad) + "\n"
    # validamos si existe el archivo de reporte
    if(path.exists(reporte)):
        # id carro - timestamp - tipo - dirección - distancia
        # DATE - TIME - N de vehículos livianos - N de vehículos medianos (2 ejes) - N de vehículos pesados (>= 3 ejes) - Velocidad Promedio (km/h) 
        f = open(reporte, "a")
        f.write(linea)
        f.close()
    else :
        f = open(reporte, "a")
        f.write("DATE;TIME;Tipo de Vehículo;Velocidad Promedio (km/h)\n")
        f.write(linea)
        f.close()

# Función para encontrar el centro del vehículo
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

# Función para contar vehículos
def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Encontrar la posición actual del vehículo
    if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list:
            t0 = time.time()
            time_cero.append(t0)
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            t0 = time.time()
            time_cero.append(t0)
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            index = temp_down_list.index(id)
            t0 = time_cero[index] 
            print("T0 = ", t0)
            tf = time.time() - t0
            print("Tf = ", tf)
            temp_down_list.remove(id)
            time_cero.remove(t0)
            up_list[index] = up_list[index]+1
            save_data(id, typeslist[index], tf)

    elif iy > down_line_position:
        if id in temp_up_list:
            index = temp_up_list.index(id)
            t0 = time_cero[index]
            print("T0 = ", t0)
            tf = time.time() - t0
            print("Tf = ", tf)
            temp_up_list.remove(id)
            time_cero.remove(t0)
            down_list[index] = down_list[index] + 1
            save_data(id, typeslist[index], tf)
    
    # Draw circle in the middle of the rectangle
    #cv2.circle(img, center, 2, colors[index], -1)  # end here
    #print(up_list, down_list)



def detect():
    weights = 'yolov7-tiny.pt'
    #source = "rtsp://admin:ALSpassword@192.168.1.64:554/Streaming/channels/2/"
    source = "/home/vision/Videos/cars.mp4"
    imgsz = 320
    trace = True
    view_img = True
    device = 'cpu'
    conf_thres = 0.60 # porcentaje de acertividad minima
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    save_conf = True
    webcam = False

    # Cargamos el modelo de detección
    model = attempt_load(weights, map_location=device)  # FP32 model
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

    # Obtenemos nombres y colores
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()
    
    fecha = str(datetime.date.today())
    now = datetime.datetime.now()
    tiempo = now.strftime("%H:%M:%S")

    print("[" + fecha + "] - " + tiempo + " - INICIO DETECCIÓN DE VEHÍCULOS")
    
    for path, img, im0s, iimg in dataset:
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
                """ print(im0s[i].copy().shape, " i = ", i)
                imgf = im0s[i].copy()[0:360,0:200] """
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            hh, ww, cc = im0.shape
            hhh = int(hh/4)
            """ print(hhh)
            imgf = im0[0+hhh:hh-hhh,::] """
            
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain 
            
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
                        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1]) + hhh) , (int(xyxy[2]), int(xyxy[3]) + hhh), colors[int(cls)], 2, lineType=cv2.LINE_AA)
                        detection.append([int(xyxy[0]), int(xyxy[1]), (int(xyxy[2])-int(xyxy[0])), (int(xyxy[3])-int(xyxy[1])), required_class_index.index(int(cls))])
            
                boxes_ids = tracker.update(detection)

                for box_id in boxes_ids:
                    x, y, w, h, id, _= box_id
                    cv2.putText(im0, str(id), (x, y - 15 + hhh), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    count_vehicle(box_id, im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            
            # Stream results
            if view_img:
                # Draw the crossing lines
                cv2.line(im0, (0, middle_line_position + hhh), (ww, middle_line_position + hhh), (255, 0, 255), 2)
                cv2.line(im0, (0, up_line_position + hhh), (ww, up_line_position + hhh), (0, 0, 255), 2)
                cv2.line(im0, (0, down_line_position + hhh), (ww, down_line_position + hhh), (0, 0, 255), 2)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    #print(f'Done. ({time.time() - t0:.3f}s)')
    print("[" + fecha + "] - " + tiempo + " TERMINÓ EL PROGRAMA")


if __name__ == '__main__':
    
    with torch.no_grad():
        detect()
