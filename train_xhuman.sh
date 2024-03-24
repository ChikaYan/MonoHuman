# To train on xhuman dataset
# first modify configs/monohuman/xhuman/xhuman.yaml by replacing all occurrence of '00019' to target xhuman sequence name

# Then, replace XHUMAN_DATA_PATH in core/data/dataset_args.py to path to xhuman dataset

# Finally, repalce SMPLX_PKL_PATH in /home/tw554/MonoHuman/core/data/monohuman/xhuman.py to path to SMPLX folder

# then training code can be run:

python trian.py --cfg ./configs/monohuman/xhuman/xhuman.yaml resume False