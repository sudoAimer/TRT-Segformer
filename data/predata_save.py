import torch
import numpy as np
import onnxruntime as ort
import os
import cv2
import glob

onnx_path = '/root/trt/onnx/segformer.b1.1024.1024.city.160k.onnx'
img_path = '/root/trt/data/png/'
npz_path = '/root/trt/data/npy/'
img_save_path = '/root/trt/data/onnx_save/'


def transfrom_img(imgPath):
    img=cv2.imread(imgPath)
    img = cv2.resize(img,(1024,1024))
    mean,std= [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    final=np.transpose((img-mean)/std,(2,0,1))
    final=np.expand_dims(final,axis=0).astype(np.float32)
    return final

def saveOutImg(a,savtePath):
    m = a.astype(np.float32)
   
    m = ((m-np.min(m))/(np.max(m)-np.min(m))*255).astype(np.uint8)
    m = np.transpose(m.reshape(m.shape[1:]),(1,2,0))
    cv2.imwrite(savtePath,m)



if __name__=='__main__':
    model = ort.InferenceSession(onnx_path)
    for image in glob.glob(img_path + '/*.png'):
        img_name = os.path.basename(image)

        input = transfrom_img(image)
        output = model.run(None, {'img': input})

        saveOutImg(output[0], os.path.join(img_save_path, img_name.split('.')[0] + '.png'))




