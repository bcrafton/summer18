
python tf_mnist_fc.py      --gpu 0 --alpha 0.01  --epochs 100 > tf_mnist_fc &
python tf_mnist_conv.py    --gpu 0 --alpha 0.001 --epochs 100 > tf_mnist_conv &
python tf_cifar10_fc.py    --gpu 0 --alpha 0.01  --epochs 100 > tf_cifar10_fc &
python tf_cifar10_conv.py  --gpu 0 --alpha 0.001 --epochs 100 > tf_cifar10_conv &
python tf_cifar100_fc.py   --gpu 0 --alpha 0.01  --epochs 100 > tf_cifar100_fc &
python tf_cifar100_conv.py --gpu 0 --alpha 0.001 --epochs 100 > tf_cifar100_conv &
