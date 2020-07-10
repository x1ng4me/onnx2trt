import engine as eng
from onnx import ModelProto
import tensorrt as trt
import onnx
import math
import cv2
import os
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine_name = '../models/onnx/centerface.trt'
onnx_path = "../models/onnx/centerface.onnx"
def onnx_2_trt(engine_path, onnx_path):
    batch_size = 10

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    
    shape = [batch_size , d0, d1 ,d2]
    print("shape",shape)
    engine = eng.build_engine(onnx_path, shape= shape)
    eng.save_engine(engine, engine_path)

def check_onnx_model(model_path):
    model = onnx.load_model(model_path)
    d = model.graph.input[0].type.tensor_type.shape.dim
    print("check_onnx_model input dim:",d)
    onnx.checker.check_model(model)

def onnx_2_scale_trt(batch_size, input_size, engine_path, onnx_path):
    model = onnx.load_model(onnx_path)

    onnx.checker.check_model(model)
    d = model.graph.input[0].type.tensor_type.shape.dim
    print(d)
    rate = (int(math.ceil(input_size[0]/d[2].dim_value)),int(math.ceil(input_size[1]/d[3].dim_value)))
    
    # d[0].dim_value = batch_size
    d[2].dim_value *= rate[0]
    d[3].dim_value *= rate[1]

    print("d out:",d)
    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
        print(d)
        # d[0].dim_value = batch_size
        d[2].dim_value  *= rate[0]
        d[3].dim_value  *= rate[1]

    onnx.save_model(model,engine_path)
    check_onnx_model(engine_path)

def get_engine(max_batch_size=10, onnx_file_path="", engine_file_path="", fp16_mode=False, int8_mode=False, save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            assert network is not None
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            
            engine = builder.build_cuda_engine(network)
            assert engine is not None
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)

batch_size = 20 
input_size = (736, 1280)
# onnx_2_trt(engine_name, onnx_path)
img = cv2.imread("../prj-python/000388.jpg")
# print("im size:", img.shape)
# resized = img[0:input_size[0], 0:input_size[1]]
# print("resized size:", resized.shape)
resized = cv2.resize(img, (input_size[1], input_size[0]), interpolation = cv2.INTER_AREA)
cv2.imwrite("../prj-python/000388_resize.png", resized)


resize_engine_name = "../models/onnx/centerface_scale.trt"
resize_model_name = '../models/onnx/centerface_scale.onnx'
onnx_path = "../models/onnx/centerface.onnx"

if os.path.exists(resize_model_name):
    os.remove(resize_model_name)
if os.path.exists(resize_engine_name):
    os.remove(resize_engine_name)

onnx_2_scale_trt(batch_size, input_size, resize_model_name, onnx_path)
get_engine(onnx_file_path=resize_model_name, engine_file_path=resize_engine_name, save_engine= True, max_batch_size = batch_size)
# onnx_2_trt(resize_engine_name, resize_model_name)

# onnx_2_trt(engine_name, onnx_path)
# get_engine(onnx_file_path=resize_model_name, engine_file_path=resize_engine_name, save_engine= True)

# img = cv2.imread("../prj-python/000388.jpg")
# dim = (768,1024)
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imwrite("../prj-python/000388_resize.png", resized)