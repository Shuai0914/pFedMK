# Due to file size limitations, only the MNIST dataset was included.

nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 10 -data mnist-0.1-npz -m cnn -algo pFedMK -did 6 -lam 5 > result-mnist-0.1-npz.out 2>&1 &