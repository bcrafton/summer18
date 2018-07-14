#python lif_group.py --examples 5 > ./results/ex5
#python eval.py --spks ./results/spks_5.npy --labels ./results/labels_5.npy > ./results/eval5

#python lif_group.py --examples 1000 > ./results/ex1000
#python eval.py --spks ./results/spks_1000.npy --labels ./results/labels_1000.npy > ./results/eval1000

#python lif_group.py --examples 10000 > ./results/ex10000
#python eval.py --spks ./results/spks_10000.npy --labels ./results/labels_10000.npy > ./results/eval10000

#python lif_group_train.py --examples 100 > ./results/train_ex100 &
#python lif_group_train.py --examples 1000 > ./results/train_ex1000 &
#python lif_group_train.py --examples 10000 > ./results/train_ex10000 &
#python lif_group_train.py --examples 50000 > ./results/train_ex50000 &

#python lif_group_test.py --examples 100 > ./results/test_ex100 &
#python lif_group_test.py --examples 1000 > ./results/test_ex1000 &
#python lif_group_test.py --examples 10000 > ./results/test_ex10000 &
#python lif_group_test.py --examples 50000 > ./results/test_ex50000 &

#python lif_group_train.py --examples 100 &
#python lif_group_train.py --examples 1000 &
#python lif_group_train.py --examples 10000 &
#python lif_group_train.py --examples 50000 &

#python lif_group_test.py --examples 100 &
#python lif_group_test.py --examples 1000 &
#python lif_group_test.py --examples 10000 &
#python lif_group_test.py --examples 50000 &

python lif_group_train.py --examples 10000 &
python lif_group_test.py --examples 10000 &
