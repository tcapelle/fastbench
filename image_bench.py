from fastbench.train_imagenette import main as image_main

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    lr:    Param("Learning rate", float)=1e-2,
    size:  Param("Size (px: 128,192,256)", int)=128,
    epochs:Param("Number of epochs", int)=10,
    bs:    Param("Batch size", int)=64,
    arch:  Param("Architecture", str)='xresnet50',
    act_fn:Param("Activation function", str)='ReLU',
    fp16:  Param("Use mixed precision training", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    
image_main(gpu=gpu,lr=lr, size=size, epochs=epochs, bs=bs, 
           arch=arch, act_fn=act_fn, fp16=fp16, runs=runs)