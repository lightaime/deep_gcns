#!/bin/bash          
MODEL="ResGCN-28"

python train.py --log_dir $MODEL/log1 --test_area 1 --num_layers 28 --num_neighbors 16 --stochastic_dilation --skip_connect 'residual' --edge_lay 'dilated' --num_filters 64 --gcn 'edgeconv'
python train.py --log_dir $MODEL/log2 --test_area 2 --num_layers 28 --num_neighbors 16 --stochastic_dilation --skip_connect 'residual' --edge_lay 'dilated' --num_filters 64 --gcn 'edgeconv'
python train.py --log_dir $MODEL/log3 --test_area 3 --num_layers 28 --num_neighbors 16 --stochastic_dilation --skip_connect 'residual' --edge_lay 'dilated' --num_filters 64 --gcn 'edgeconv'
python train.py --log_dir $MODEL/log4 --test_area 4 --num_layers 28 --num_neighbors 16 --stochastic_dilation --skip_connect 'residual' --edge_lay 'dilated' --num_filters 64 --gcn 'edgeconv'
python train.py --log_dir $MODEL/log5 --test_area 5 --num_layers 28 --num_neighbors 16 --stochastic_dilation --skip_connect 'residual' --edge_lay 'dilated' --num_filters 64 --gcn 'edgeconv'
python train.py --log_dir $MODEL/log6 --test_area 6 --num_layers 28 --num_neighbors 16 --stochastic_dilation --skip_connect 'residual' --edge_lay 'dilated' --num_filters 64 --gcn 'edgeconv'
