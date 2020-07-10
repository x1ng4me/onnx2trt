import numpy as np
import cupy as cp
import os
import cv2
import glob
import torch
import datetime
import torchvision
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from functools import partial
# from nms import nms 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
    
class CenterFace(object):
    def __init__(self, batch_size,  landmarks=True):
        self.landmarks = landmarks
        self.trt_logger = trt.Logger()  # This logger is required to build an engine
        
        f = open("../models/onnx/centerface_scale.trt", "rb")
        runtime = trt.Runtime(self.trt_logger)

        print("__init__ start")
        self.net = runtime.deserialize_cuda_engine(f.read())
        self.engine_height, self.engine_width = (736, 1280)
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.engine_height, self.engine_width, 1, 1

        self.context = self.net.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.net)
        print("__init__ done")
        self.batch_size = batch_size
        self.shape_of_output = [(self.batch_size, 1, int(self.img_h_new / 4), int(self.img_w_new / 4)),
                           (self.batch_size, 2, int(self.img_h_new / 4), int(self.img_w_new / 4)),
                           (self.batch_size, 2, int(self.img_h_new / 4), int(self.img_w_new / 4)),
                           (self.batch_size, 10, int(self.img_h_new / 4), int(self.img_w_new / 4))]
        # Do inference


    def __call__(self, imgs, threshold=0.5):
        size = imgs.shape
        
        h, w = size[2:4]
        assert h == self.engine_height
        assert w == self.engine_width
        deal_batch = size[0]
        # Allocate buffers for input and output
        inputs, outputs, bindings, stream = self.inputs, self.outputs, self.bindings, self.stream

        # Load data to the buffer
        inputs[0].host = torch.flatten(imgs).numpy()
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=deal_batch)  # numpy data
        end = datetime.datetime.now()

        heatmap, scale, offset, lms = [output.reshape(shape) for output, shape in zip(trt_outputs, self.shape_of_output)]
        return self.postprocess(heatmap, lms, offset, scale, threshold)[:deal_batch]

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        dets = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        rlts = [self.decodeImp((heatmap, scale, offset, landmark, size, threshold), i) for i in  range(heatmap.shape[0])]
        return list(rlts)
    
    def decodeImp(self, input_arg, i):
        heatmap, scale, offset, landmark, size, threshold = input_arg
        heatmap = np.squeeze(heatmap[i])
        scale0, scale1 = scale[i, 0, :, :], scale[i, 1, :, :]
        offset0, offset1 = offset[i, 0, :, :], offset[i, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
       
        boxes, lms = np.empty(shape=[0, 5], dtype=np.float32), np.empty(shape=[0, 10], dtype=np.float32)
        if len(c0) > 0:
            s0, s1 = np.exp(scale0[c0,c1])*4,np.exp(scale1[c0, c1])*4
            o0, o1 = offset0[c0,c1], offset1[c0,c1]
            s = heatmap[c0,c1]
            x1, y1 = np.maximum(0, (c1 + o1+0.5)*4-s1/2),np.maximum(0, (c0 + o0+0.5)*4-s0/2)
            x1, y1 = np.minimum(x1, size[1]), np.minimum(y1, size[0])
            x2, y2 = np.minimum(x1 + s1, size[1]), np.minimum(y1 + s0, size[0])

            if self.landmarks:
                lms = []
                for j in range(5):
                    lms.append(np.multiply(landmark[i, j * 2 + 1, c0, c1], s1)+ x1)
                    lms.append(np.multiply(landmark[i, j * 2, c0, c1] , s0) + y1)
                lms = np.vstack(lms).T

            boxes = np.vstack((x1, y1, x2, y2,s)).T 
            keep = torchvision.ops.nms(torch.Tensor(boxes[:, :4]), torch.Tensor(boxes[:, 4]), 0.3).numpy()
            # print("keep", len(keep))

            # print("keep", keep)
            boxes = boxes[keep, :]
            boxes[:, 0:4:2], boxes[:, 1:4:2] = boxes[:, 0:4:2] / self.scale_w, boxes[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        return boxes, lms

def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的RGB图片数据
    '''
 
    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    rgb_image = cv2.resize(rgb_image, dsize=(resize_width, resize_height))
    rgb_image = rgb_image.transpose(2, 0, 1)
    rgb_image = np.asanyarray(rgb_image)
    rgb_image = torch.Tensor(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image


class VideoFrameDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.size = size
        self.root_dir = root_dir
        self.img_paths = sorted(glob.glob(os.path.join(self.root_dir, "*")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        image = read_image(img_path, self.size[0], self.size[1])

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)


centerface = CenterFace(20)

preprocess = transforms.Compose([
    transforms.ToTensor()
])
size = (736, 1280)
path = "../../data/test_batch/"
dataset = VideoFrameDataset(path,size)
dataloader = DataLoader(dataset, batch_size=20, num_workers = 10,  shuffle=True)

for step, (batch_image, image_paths) in tqdm(enumerate(dataloader)):
    dets = centerface(batch_image, threshold=0.35)
    for det, img_path, img in zip(dets, image_paths, batch_image):
        save_debug_date(img_path, det, img)


    # print(np_data.shape)
    # print(np_data)
    # break