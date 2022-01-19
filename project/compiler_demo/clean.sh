#!/bin/bash
echo "clean forward files"
rm -rf relayIR/test.txt
rm -rf relayIR/all_weight2.dat
rm -rf quan_pth/Epoch1-YOLOV4_quantization_post_jit.pth
rm -rf quan_pth/Epoch1-YOLOV4_quantization_post_save.pth
