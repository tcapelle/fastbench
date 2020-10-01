import torch
from fastscript import *
from fastbench.utils import show_install
from fastbench.vision import train_imagenette
from fastbench.tabular import train_tabular
from fastbench.imdb import train_imdb


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
    train_imagenette(gpu=gpu,epochs=epochs,fp16=0,bs=64,runs=runs)
    train_imagenette(gpu=gpu,epochs=epochs,fp16=1,bs=128,runs=runs)
    
    #run imdb
    print('\nBenchmarking IMDB')
    train_imdb(gpu=gpu,epochs=epochs,fp16=0,bs=64,runs=runs)
    train_imdb(gpu=gpu,epochs=epochs,fp16=1,bs=128,runs=runs)
    
    #run tabular
    print('\nBenchmarking Tabular')
    train_tabular(gpu=gpu,epochs=epochs,fp16=0,bs=64,runs=runs)
    train_tabular(gpu=gpu,epochs=epochs,fp16=1,bs=128,runs=runs)