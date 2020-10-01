import torch
from fastscript import *
from fastbench.utils import show_install

from fastbench.train_imagenette import main as image_main
from fastbench.train_imdbclassifier import main as imdb_main
from fastbench.train_wt2 import main as wt2_main


"""Compute image benchmarks on imagenette"""

welcome_str="""Welcome to fastbench! your friendly benchmark for GPU using fastai.\n"""

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    epochs:Param("Number of epochs", int)=5,
    runs:  Param("Number of times to repeat training", int)=1,
):
    print(welcome_str)
    show_install()
    print('==================')
    if gpu is not None: 
        print(f'\nRunning bench on: GPU: {torch.cuda.get_device_name(gpu)}')
        
    #run image benchsmarks
    print('\nBenchmarking Imagenette')
    image_main(gpu=gpu,epochs=epochs,fp16=0,bs=64,runs=runs)
    image_main(gpu=gpu,epochs=epochs,fp16=1,bs=128,runs=runs)
    
    #run imdb
    print('\nBenchmarking IMDB')
    imdb_main(gpu=gpu,epochs=epochs,fp16=0,bs=64,runs=runs)
    imdb_main(gpu=gpu,epochs=epochs,fp16=1,bs=128,runs=runs)
    
    #run tabular
    print('\nBenchmarking Tabular')
    imdb_main(gpu=gpu,epochs=epochs,fp16=0,runs=runs)
    imdb_main(gpu=gpu,epochs=epochs,fp16=1,runs=runs)