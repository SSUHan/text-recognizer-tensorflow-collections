# text-recognizer-tensorflow-collections

Reproduce various text recognition model for OCR proposed in papers with TensorFlow implementation

## Papers & Code

relative with `text recognition` task

- [2018-ECCV] Synthetically Supervised Feature Learning for Scene Text Recognition [`paper`](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.pdf) (keywords: SSFL)
- [2018-CVPR] AON: Towards Arbitrarily-Oriented Text Recognition [`paper`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_AON_Towards_Arbitrarily-Oriented_CVPR_2018_paper.pdf) (keywords: AON)
- [2017-CVPR] **Focusing Attention: Towards Accurate Text Recognition in Natural Images** [`paper`](https://arxiv.org/abs/1709.02054) (keywords: FAN)
- [2017-NIPS] Gated Recurrent Convolution Neural Network for OCR [`paper`](https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) (keywords: GRCNN)
- [2017-PAMI] **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition** [`paper`](https://arxiv.org/abs/1507.05717) (keywords: CRNN)
- [2016-CVPR] Robust Scene Text Recognition with Automatic Rectification [`paper`](https://arxiv.org/abs/1603.03915) (keywords:  RARE)
- [2016-IEEE] Recursive Recurrent nets with Attention Modeling for ocr in the wild [`paper`](https://arxiv.org/abs/1603.03101) (keywords: R2AM)



## Benchmark Dataset

|       | IIIT5k_50 | IIIT5k_1K | IIIT5k_None | SVT_50 | SVT_None | IC03_50 | IC03_full | IC03_50K | IC03_None |   IC13_857_None   | IC13_1015_None | IC15_None |
| ----- | :-------: | :-------: | :---------: | :----: | :------: | :-----: | :-------: | :------: | :-------: | :---------------: | :------------: | :-------: |
| CRNN  |   97.6    |   94.4    |    78.2     |  96.4  |   80.8   |  98.7   |   97.6    |   95.5   |   89.4    | 86.7 (from SSFL)  |      86.7      |     -     |
| SSFL  |   97.3    |   96.1    |    89.4     |  96.8  |   87.1   |  98.1   |   97.5    |    -     |   94.7    |       94.0        |       -        |     -     |
| FAN   |   99.3    |   97.5    |    87.4     |  97.1  |   85.9   |  99.2   |   97.3    |    -     |   94.2    | 93.9 (from SSFL)  |      93.3      |   70.6    |
| GRCNN |   98.0    |   95.6    |    80.8     |  96.3  |   81.5   |  98.8   |   97.8    |    -     |   91.2    |         -         |       -        |     -     |
| RARE  |   96.2    |   93.8    |    81.9     |  95.5  |   81.9   |  98.3   |   96.2    |    -     |   90.1    |    88.6 / 87.5    |       -        |     -     |
| R2AM  |   96.8    |   94.4    |    78.4     |  96.3  |   80.7   |  97.9   |   97.0    |    -     |   88.7    | 90.0 ( from SSFL) |      90.0      |     -     |





### How to train

command example : 
```
python main.py --gpu 0 --model_name CRNN --testset ic13_857,ic13_1015,ic03_867 --optimizer RMSProp --loss ctc
```

```
python main.py --gpu 1 --model_name CRNN --testset ic13_857,ic13_1015,ic03_867 --optimizer Adam --loss ctc
```



## References

- [chongyangtao-github](https://github.com/chongyangtao/Awesome-Scene-Text-Recognition)