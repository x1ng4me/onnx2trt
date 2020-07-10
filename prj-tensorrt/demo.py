import cv2
import sys
import scipy.io as sio
import os
import datetime
from centerface import CenterFace
import tensorrt as trt

#sudo nvidia-docker run -it --ipc=host --name tensorrt7  adujardin/tensorrt-trtexec:7.0  --onnx=centerface.onnx  --saveEngine=centerface_engine.trt --useCudaGraph
def test_image_tensorrt():
    frame = cv2.imread('../prj-python/000388.jpg')
    
    landmarks = True
    print("start:",frame.shape)
    centerface = CenterFace()

    begin = datetime.datetime.now()
    if landmarks:
        dets = centerface(frame, threshold=0.35)
        print("count = ", len(dets))
    else:
        dets = centerface(frame, threshold=0.35)
    dets = dets[0]

    end = datetime.datetime.now()
    print("test_image_tensorrt times = ", end - begin)
    for det in dets[0]:
        # print(det)
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in dets[1]:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imwrite("000388_out2.png", frame)
    print("sucdess!")

if __name__ == '__main__':
    # import numpy as np
    # import cupy as cp
    # import time

    # ### Numpy and CPU
    # s = time.time()
    # x_cpu = np.ones((1000,1000,3))
    # e = time.time()
    # print(e - s)
    # ### CuPy and GPU
    # s = time.time()
    # x_gpu = cp.ones((1000,1000,3))
    # cp.cuda.Stream.null.synchronize()
    # e = time.time()
    # print(e - s)

    # ### Numpy and CPU
    # s = time.time()
    # x_cpu *= 5
    # e = time.time()
    # print(e - s)
    # ### CuPy and GPU
    # s = time.time()
    # x_gpu *= 5
    # cp.cuda.Stream.null.synchronize()
    # e = time.time()
    # print(e - s)

    test_image_tensorrt()
    # network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    # network_creation_flag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # with trt.Builder(TRT_LOGGER) as builder, \
    #     builder.create_network(network_creation_flag) as network, \
    #     trt.OnnxParser(network, TRT_LOGGER) as parser:
            
    #     # Fill network atrributes with information by parsing model
    #     with open("../models/onnx/centerface.onnx", "rb") as f:
    #         if not parser.parse(f.read()):
    #             print('ERROR: Failed to parse the ONNX file: ')
    #             for error in range(parser.num_errors):
    #                 print(parser.get_error(error))
    #             sys.exit(1)