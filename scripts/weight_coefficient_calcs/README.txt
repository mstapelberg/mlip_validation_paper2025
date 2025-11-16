This code was ran with:

python suggest_loss_weights.py   --train data/exploit_q-97.5_rmax5.50_lmax1_layers1_mlp256_seed42_train.xyz   --val data/exploit_q-97.5_rmax5.50_lmax1_layers1_mlp256_seed42_val.xyz   --stress-units auto   --use-config-aware   --weight-p 0.20 --weight-s 0.80   --allow-pressure-for-defects   --target E=1,F=1,S=1   --loss-types energy=mse,force=huber,stress=huber   --normalize-to energy --json-out loss_weights.json

For access to the train and val data, please email myless@mit.edu
