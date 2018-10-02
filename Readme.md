# Text Recognzier Benchmark

## How to train
command example : 
```
python main.py --gpu 0 --model_name CRNN --testset ic13_857,ic13_1015,ic03_867 --optimizer RMSProp --loss ctc
```

```
python main.py --gpu 1 --model_name CRNN --testset ic13_857,ic13_1015,ic03_867 --optimizer Adam --loss ctc
```