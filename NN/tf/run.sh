
python tf_mnist_fc.py      --gpu 0 --alpha 0.01  --epochs 100 > ./res/tf_mnist_fc &
python tf_mnist_conv.py    --gpu 0 --alpha 0.005 --epochs 200 > ./res/tf_mnist_conv &
python tf_cifar10_fc.py    --gpu 1 --alpha 0.01  --epochs 100 > ./res/tf_cifar10_fc &
python tf_cifar10_conv.py  --gpu 2 --alpha 0.005 --epochs 200 > ./res/tf_cifar10_conv &
python tf_cifar100_fc.py   --gpu 1 --alpha 0.01  --epochs 100 > ./res/tf_cifar100_fc &
python tf_cifar100_conv.py --gpu 3 --alpha 0.005 --epochs 200 > ./res/tf_cifar100_conv &
