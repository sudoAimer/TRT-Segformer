import os
import glob
import cv2
from PIL import Image
import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
#https://documen.tician.de/pycuda/driver.html
import pycuda.autoinit
import sys
import time
import ctypes


def np_normalize(data, mean, std):
    # transforms.ToTensor, transforms.Normalize的numpy 实现
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    if mean.ndim == 1:
        mean = np.reshape(mean, (-1, 1, 1))
    if std.ndim == 1:
        std = np.reshape(std, (-1, 1, 1))
    _max = np.max(abs(data))
    _div = np.divide(data, _max)  # i.e. _div = data / _max
    _sub = np.subtract(_div, mean)  # i.e. arrays = _div - mean
    arrays = np.divide(_sub, std)  # i.e. arrays = (_div - mean) / std
    arrays = np.transpose(arrays, (2, 0, 1))
    return arrays


def imread(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB').resize((1024, 1024))
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3))
    std  = np.array([0.229, 0.224, 0.225]).reshape((1, 3))
    img = np_normalize(img, mean, std)
    img = np.expand_dims(img, 0)
    return img.astype(np.float32)


class Calibrator(trt.IInt8EntropyCalibrator2):
    '''calibrator
        IInt8EntropyCalibrator2
        IInt8LegacyCalibrator
        IInt8EntropyCalibrator
        IInt8MinMaxCalibrator
    '''
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)       
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        # print(self.cache_file)
        stream.reset()
        

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):

        batch = self.stream.next_batch()
        if not batch.size:  
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print(f"[INFO] Using calibration cache to save time: {self.cache_file}")
                return f.read()

    def write_calibration_cache(self, cache): 
        with open(self.cache_file, "wb") as f:
            print(f"[INFO] Caching calibration data for future use: {self.cache_file}")
            f.write(cache)


class DataLoader:
    def __init__(self,calib_img_dir="/root/trt/data/calib_data/", batch=1,batch_size=1):
        
        self.index = 0
        # self.length = batch
        self.length = 10
        self.batch_size = batch_size
        self.calib_img_dir = calib_img_dir

        self.img_list = glob.glob(os.path.join(self.calib_img_dir, "*.png"))
        print(f'[INFO] found all {len(self.img_list)} images to calib.')
        assert len(self.img_list) >= self.batch_size * self.length, '[Error] {} must contains more than {} images to calib'.format(self.calib_img_dir,self.batch_size * self.length)
        self.calibration_data = np.zeros((self.batch_size, 3, 1024, 1024), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), '[Error] Batch not found!!'
                img = imread(self.img_list[i + self.index * self.batch_size])
                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TensorRT INT8 Quant.')
    parser.add_argument('--onnx_model_path', type=str , default='/root/trt/onnx/sim.segformer.b1.1024x1024.city.160k_v1.onnx', help='ONNX Model Path')    
    parser.add_argument('--engine_model_path', type=str , default='/root/trt/plan/sim_fp16_segformer_b1_1024_1024_city_160k_v1_int8.plan', help='TensorRT Engine File')
    parser.add_argument('--calib_img_dir', type=str , default='/root/trt/data/calib_data/', help='Calib Image Dir')   
    parser.add_argument('--calibration_table', type=str,default="/root/trt/calib/segformer_calibration_test.cache", help='Calibration Table')
    parser.add_argument('--batch', type=int,default=10, help='Number of Batch: [total_image/batch_size]')  # 30660/batch_size
    parser.add_argument('--batch_size', type=int,default=10, help='Batch Size')

    parser.add_argument('--fp16', action="store_true", help='Open FP16 Mode')
    parser.add_argument('--int8', action="store_true", help='Open INT8 Mode')

    args = parser.parse_args()
    ctypes.CDLL("/root/trt/soFile/LayerNorm.so")
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, "")

    if os.path.isfile(args.engine_model_path):
        with open(args.engine_model_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.flags = 1 << int(trt.BuilderFlag.INT8) | 1 << int(trt.BuilderFlag.FP16)

        calibration_stream = DataLoader(calib_img_dir=args.calib_img_dir,batch=args.batch,batch_size=args.batch_size)
        config.int8_calibrator = Calibrator(calibration_stream, args.calibration_table)
        config.max_workspace_size = 23 << 30
        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(args.onnx_model_path):
            print("Failed finding onnx file!")
            exit()
        print("Succeeded finding onnx file!")
        with open(args.onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        inputTensor = network.get_input(0)
        # profile.set_shape(inputTensor.name, (1, 3, 1024, 1024), (4, 3, 1024, 1024), (8, 3, 1024, 1024))
        config.add_optimization_profile(profile)

        engineString = builder.build_serialized_network(network, config)

        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open("segformer_test_int8.plan", 'wb') as f:
            f.write(engineString)

        # engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
