

python cifar10_dfa_nn.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.001 --gpu 0 &
python cifar10_dfa_nn.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0005 --gpu 1 &
python cifar10_dfa_nn.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0001 --gpu 2 &
python cifar10_dfa_nn.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.00005 --gpu 3 &

wait

python cifar10_nn_args.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.001 --gpu 0 &
python cifar10_nn_args.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0005 --gpu 1 &
python cifar10_nn_args.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0001 --gpu 2 &
python cifar10_nn_args.py --layers 3072 4096 1024 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.00005 --gpu 3 &
