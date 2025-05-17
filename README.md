# Self-Supervised Learning for End-to-End Particle Reconstruction for the CMS Experiment


## Project aim
The aim of the project is to investigate on the use of an I-JEPA architecture for the self-supervised pre-training stage on unlabeled data from the CMS experiment. The pre-training stage is validated fine-tuning by linear probing the frozen pre-trained models on a downstream binary particle classification task.


## Code
You can run the following line of code to start the training of a model with a given configuration:
```bash
python main.py --fname /configs/vit_b_14.yaml devices cuda:0
```
Moreover, specifying the number of layers to unfreeze in the config file, you can linear probe or fine tune a model with the line:
```bash
python linear_probing.py  --fname configs/probing_vit_b_14.yaml --devices cuda:0
```


## Results
Those are the obtained results. All scores are ROC-AUC metric. To read more about those results check out the second blogpost.

| Model Name      |  Scratch        | Linear probing | Fine-tuning        |
| --------------- | --------------- | -------------- | ------------------ |
| vit_s_14        | 0.75            | 0.732          | 0.74               |
| vit_s_9         | 0.76            | 0.731          | 0.76               |
| vit_b_14        | 0.74            | 0.737          | 0.78               |
| vit_b_9         | 0.73            | 0.738          | 0.79               |


