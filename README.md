# Implement unbiased teacher algorithm on blazeface model by pytorch

##### Data
1. The dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

2. We provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training

1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``config/config.py and tool/train_ut.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python train_ut.py
  ```

## Evaluation
### Evaluation widerface val
```
Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
python setup.py build_ext --inplace
python evaluation.py
```


## References
- [blazeface](https://github.com/zineos/blazeface)
- [unbiased teacher v2](https://github.com/facebookresearch/unbiased-teacher-v2)

