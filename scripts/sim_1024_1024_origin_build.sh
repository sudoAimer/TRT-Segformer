export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-8.6.1.6/lib/

cd ~

./TensorRT-8.6.1.6/bin/trtexec \
    --onnx=./trt/onnx/segformer.b1.1024x1024.city.160k_v1.onnx \
    --workspace=15000 \
    --saveEngine=./trt/plan/fp32_segformer_b1_1024_1024_city_160k_v1.plan \
    --verbose \
    --plugins=/root/trt/soFile/LayerNorm.so \
    > ./trt/log/plan_b1_1024_1024_fp32_v1.log 