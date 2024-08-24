# Measuring Style Similarity in Diffusion Models

![alt text](github_teaser.jpg "Generations from Stable Diffusion and corresponding matches from LAION-Styles split")

## Create and activate the environment

```
conda env create -f environment.yml
conda activate style
```

## Download the pretrained weights for the CSD model

Please download the CSD model (ViT-L) weights [here](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view?usp=sharing). 


## Download the pretrained weights for the baseline models

You need these only if you want to test the baseline numbers. For `CLIP` and `DINO`, pretrained weights will be downloaded automatically. For `SSCD` and `MoCo`, please download the weights
from the links below and put them in `./pretrainedmodels` folder.

* SSCD: [resnet50](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt)
* MoCO: [ViT-B](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar)



## Download the WikiArt dataset
WikiArt can be downloaded from [here](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk0)
or [here1](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip)

After dataset is downloaded please put `./wikiart.csv` in the parent directory of the dataset. The final directory structure should look like this:
```
path/to/WikiArt
├── wikiart
    ├── Abstract_Expressionism
        ├── <filename>.jpg
    ├── ...
└── wikiart.csv
```

Also, make sure that you add a column `path` in the `wikiart.csv` file which contains the absolute path to the image.

## Generate the embeddings

Once WikiArt dataset is set up, you can generate the CSD embeddings by running the following command. Please adjust
the `--data-dir` and `--embed_dir` accordingly. You should also adjust the batch size `--b` and number of workers `--j`
according to your machine. The command to generate baseline embeddings is same, you just need to change the `--pt_style`
with any of the following: `clip`, `dino`, `sscd`, `moco`.

```angular2html
python main_sim.py --dataset wikiart -a vit_large --pt_style csd --feattype normal --world-size 1 
--dist-url tcp://localhost:6001 -b 128 -j 8 --embed_dir ./embeddings --data-dir <path to WikiArt dataset>
--model_path <path to CSD weights>
```

## Evaluate
Once you've generated the embeddings, run the following command:

```angular2html
python search.py --mode artist --dataset wikiart --chunked --query-chunk-dir <path to query embeddings above> 
    --database-chunk-dir <path to database embeddings above> --topk 1 10 100 1000 --method IP --data-dir <path to WikiArt dataset>
```

## Train CSD on LAION-Styles

You can also train style descriptors for your own datasets. A sample code for training on LAION-styles dataset is provided below. (We will release the dataset construction files soon.)

```
export PYTHONPATH="$PWD:$PYTHONPATH"

torchrun --standalone --nproc_per_node=4 CSD/train_csd.py --arch vit_base -j 8 -b 32 --maxsize 512 --resume_if_available --eval_k 1 10 100 --use_fp16 --use_distributed_loss --train_set laion_dedup --train_path <PATH to LAION-Styles> --eval_path <PATH to WikiArt/some val set>  --output_dir <PATH to save checkpoint>
```

## Pending items

1. We will soon release the code to compute the artists' prototypical style representations and compute similarity score against any given generation. 
