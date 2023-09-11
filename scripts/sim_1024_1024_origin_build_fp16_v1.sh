export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-8.6.1.6/lib/

cd ~

./TensorRT-8.6.1.6/bin/trtexec \
    --onnx=./trt/onnx/sim.segformer.b1.1024.1024.city.160k.onnx \
    --fp16 \
    --workspace=15000 \
    --saveEngine=./trt/plan/sim_fp16_segformer_b1_1024_1024_city_160k_v1.plan \
    --verbose \
    --plugins=/root/trt/soFile/LayerNorm.so \
    > ./trt/log/sim_plan_b1_1024_1024_fp16_v1.log 
