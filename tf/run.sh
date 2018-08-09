
python mnist_dfa_nn.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.01 --gpu 0
python mnist_dfa_nn.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.005 --gpu 1
python mnist_dfa_nn.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.001 --gpu 2
python mnist_dfa_nn.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.0005 --gpu 3

python mnist_nn_args.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.01 --gpu 0
python mnist_nn_args.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.005 --gpu 1
python mnist_nn_args.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.001 --gpu 2
python mnist_nn_args.py --layers 784 500 300 100 10 --epochs 1000 --batch_size 32 --alpha 0.0005 --gpu 3

python cifar_dfa_nn.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.001 --gpu 0
python cifar_dfa_nn.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0005 --gpu 1
python cifar_dfa_nn.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0001 --gpu 2
python cifar_dfa_nn.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.00005 --gpu 3

python cifar_nn_args.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.001 --gpu 0
python cifar_nn_args.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0005 --gpu 1
python cifar_nn_args.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.0001 --gpu 2
python cifar_nn_args.py --layers 3072 2048 1024 512 256 128 10 --epochs 1000 --batch_size 32 --alpha 0.00005 --gpu 3
