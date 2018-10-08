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