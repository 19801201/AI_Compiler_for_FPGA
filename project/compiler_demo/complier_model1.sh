#!/bin/bash
echo "begin compile"
python post_trainq.py
python relayIR/create_ins.py
python relayIR/all_nets_weight_create.py

