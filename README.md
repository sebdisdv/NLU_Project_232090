# NLU_Project_232090

## Install necessary libraries

```
pip install -r requirements.txt
```

## Commands to run the project

### Single runs [SNIPS]


- python main.py --model LSTM --task snips --crf 0 --dropout 0 --runs 1 --device cuda 
- python main.py --model GRU --task snips --crf 0 --dropout 0 --runs 1 --device cuda
- python main.py --model JOINTBERT --task snips --crf 0 --dropout 0 --runs 1 --device cuda

- python main.py --model LSTM --task snips --crf 1 --dropout 1 --runs 1 --device cuda 
- python main.py --model GRU --task snips --crf 1 --dropout 1 --runs 1 --device cuda
- python main.py --model JOINTBERT --task snips --crf 1 --dropout 1 --runs 1 --device cuda

- python main.py --model LSTM --task snips --crf 1 --dropout 0 --runs 1 --device cuda 
- python main.py --model GRU --task snips --crf 1 --dropout 0 --runs 1 --device cuda
- python main.py --model JOINTBERT --task snips --crf 1 --dropout 0 --runs 1 --device cuda

- python main.py --model LSTM --task snips --crf 0 --dropout 1 --runs 1 --device cuda 
- python main.py --model GRU --task snips --crf 0 --dropout 1 --runs 1 --device cuda
- python main.py --model JOINTBERT --task snips --crf 0 --dropout 1 --runs 1 --device cuda



### Single runs [ATIS]

- python main.py --model LSTM --task atis --crf 0 --dropout 0 --runs 1 --device cuda 
- python main.py --model GRU --task atis --crf 0 --dropout 0 --runs 1 --device cuda
- python main.py --model JOINTBERT --task atis --crf 0 --dropout 0 --runs 1 --device cuda

- python main.py --model LSTM --task atis --crf 1 --dropout 1 --runs 1 --device cuda 
- python main.py --model GRU --task atis --crf 1 --dropout 1 --runs 1 --device cuda
- python main.py --model JOINTBERT --task atis --crf 1 --dropout 1 --runs 1 --device cuda

- python main.py --model LSTM --task atis --crf 1 --dropout 0 --runs 1 --device cuda 
- python main.py --model GRU --task atis --crf 1 --dropout 0 --runs 1 --device cuda
- python main.py --model JOINTBERT --task atis --crf 1 --dropout 0 --runs 1 --device cuda

- python main.py --model LSTM --task atis --crf 0 --dropout 1 --runs 1 --device cuda 
- python main.py --model GRU --task atis --crf 0 --dropout 1 --runs 1 --device cuda
- python main.py --model JOINTBERT --task atis --crf 0 --dropout 1 --runs 1 --device cuda

### Multiple runs SNIPS

- python main.py --model LSTM --task snips --crf 0 --dropout 0 --runs 4 --device cuda 
- python main.py --model GRU --task snips --crf 0 --dropout 0 --runs 4 --device cuda

- python main.py --model LSTM --task snips --crf 1 --dropout 1 --runs 4 --device cuda 
- python main.py --model GRU --task snips --crf 1 --dropout 1 --runs 4 --device cuda

- python main.py --model LSTM --task snips --crf 1 --dropout 0 --runs 4 --device cuda 
- python main.py --model GRU --task snips --crf 1 --dropout 0 --runs 4 --device cuda

- python main.py --model LSTM --task snips --crf 0 --dropout 1 --runs 4 --device cuda 
- python main.py --model GRU --task snips --crf 0 --dropout 1 --runs 4 --device cuda

### Multiple runs ATIS

- python main.py --model LSTM --task atis --crf 0 --dropout 0 --runs 4 --device cuda 
- python main.py --model GRU --task atis --crf 0 --dropout 0 --runs 4 --device cuda

- python main.py --model LSTM --task atis --crf 1 --dropout 1 --runs 4 --device cuda 
- python main.py --model GRU --task atis --crf 1 --dropout 1 --runs 4 --device cuda

- python main.py --model LSTM --task atis --crf 1 --dropout 0 --runs 4 --device cuda 
- python main.py --model GRU --task atis --crf 1 --dropout 0 --runs 4 --device cuda

- python main.py --model LSTM --task atis --crf 0 --dropout 1 --runs 4 --device cuda 
- python main.py --model GRU --task atis --crf 0 --dropout 1 --runs 4 --device cuda