# implement the multi-processing controller here
import argparse
from multiprocessing import Pool
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('data_set', type=str, help='appoint a dataset to crunch, name-train, name-test',default='synthtext-train')
parser.add_argument('Pool_size', type=int, help='pool size',default=35)
args = parser.parse_args()

p=Pool(args.Pool_size)
for i in range(args.Pool_size):
    print('Starting the %dth patch.'%i)
    subprocess.call(['python','data_churner.py',args.data_set,i,args.Pool_size,'data_cleaned/data_cleaned/tfrecord/%s_%d'%(args.data_set,i)])
p.close()
p.join()
