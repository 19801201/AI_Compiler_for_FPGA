#!/bin/bash
echo "clean forward files"
rm -rf biasscaleshift.bin 
rm -rf converted.json 
rm -rf converted.npy 
rm -rf converted.pb 
rm -rf extra_prame_coe.py 
rm -rf instruction_FPGA.bin 
rm -rf tmp_addr.txt 
rm -rf para.txt 
rm -rf q_paras/* 
rm -rf prun_model/* 
rm -rf paras/* 
rm -rf model/* 
rm -rf YOLO_quantization_post.pth 