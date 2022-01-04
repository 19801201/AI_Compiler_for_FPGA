#!/bin/bash
echo "begin compile"
echo "optlevel: "
python json2model_npy1.py
python load_weights2.py
python opt_2_5.py --optlevel $1
python post3.py
python extram_coe4.py
python extra_prame_coe.py
python inform_instruc5.py
python instruction6.py
