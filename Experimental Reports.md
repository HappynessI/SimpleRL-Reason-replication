# Experimental Reports


## Project Dependency Issues
I initially set up a virtual environment with Python 3.9 as specified in the README. When I tried to install the dependencies from requirements.txt, I ran into the following error:

```
ERROR: Could not find a version that satisfies the requirement math-verify==0.6.0 (from versions: none)
ERROR: No matching distribution found for math-verify==0.6.0
```

The logs indicated that all available versions of math-verify require Python >= 3.10.

I later found an issue where the author confirmed this, stating that Python 3.10+ is required for math-verify. For Python 3.9, the project uses Qwen's math evaluation function, which is available at: [https://github.com/hkust-nlp/simpleRL-reason/blob/v1/verl/utils/reward_score/qwen_math_eval_toolkit/grader.py#L73]

I then made the following modifications:
```
# 原26行
# from math_verify import parse, verify  # 注释掉，因为Python 3.9不支持

from .qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal

# 原120行
def hf_verify_with_try(gold, target):

    try:

        # 使用Qwen的math_equal函数替代math_verify的parse和verify

        return qwen_math_equal(prediction=target, reference=gold)

    except Exception as e:

        print(f"Gold: {gold} Target: {target} Error: {str(e)}")

        return False
```

## Validation Dataloader Issue
I also discovered a severe problem during validation: the original code at line 483 hardcodes the `batch_size` for the validation dataloader to the entire size of the validation set (`len(self.val_dataset)`), instead of using the configuration I provided in my script.

【Original Code】
```
self.val_dataloader = DataLoader(dataset=self.val_dataset,

                                 batch_size=len(self.val_dataset),

                                 shuffle=True,

                                 drop_last=True,

                                 collate_fn=collate_fn)
                                                          
```
【Modified Code】
```
self.val_dataloader = DataLoader(dataset=self.val_dataset,

                                 batch_size=self.config.data.val_batch_size,

                                 shuffle=True,

                                 drop_last=True,

                                 collate_fn=collate_fn)
```

## Memory Leak Issues
During training, I frequently encountered actor die errors, even though the server was equipped with around 480GB of RAM.

By strategically inserting ray.gc() at memory-intensive stages, I managed to partially mitigate the pressure from this apparent memory leak. However, this was not a complete solution, and I was still forced to reduce the checkpoint saving frequency.

My suspicion is that the problem stems from an outdated vllm version, which lags significantly behind the version required by the official verl repository. Ultimately, I decided to clone the official verl repository, customize it, and run my training from there.

## Qwen2.5-Math-1.5B Experiment Log

I had previously run experiments with Qwen2.5-1.5B-Instruct, but the results were not good, so I did not save the experimental parameters from that run. This current experiment (using Qwen2.5-Math-1.5B) is the one with the best-preserved parameters and results.

### Experimental Environment

- System RAM: 480GB
- GPU Resources: 2x L20

### Experimental Parameters - V1


```
# Default values
TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=128
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=128
# per GPU
PPO_MICRO_BATCH_SIZE=2
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=64
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=2
DATASET_NAME=simplelr_qwen_level3to5
ROLLOUT_GPU_MEMORY_UTIL=0.6
MODEL_NAME=Qwen2.5-Math-1.5B
SAVE_FREQ=25
TEST_FREQ=55
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=512
REMOVE_PREVIOUS_CKPT=False
```

Under this set of parameters, the first training run crashed at step_95. According to the wandb logs, the response clip_ratio was excessively high.

<img width="946" height="442" alt="image" src="https://github.com/user-attachments/assets/be144a2b-52bd-4afc-be31-be5b13067d41" />

I found that the response truncation was severe: the mean clip_ratio was around 23%, with 20-28% of samples hitting the 1024 token limit. Normally, this ratio should be less than 5%. In a math reasoning task, a truncated response is highly likely to be an incomplete solution, which severely impacts training quality.

Therefore, this experiment was abandoned.

### Reward Type Selection
In the SimpleRL-Reason repository, the file simpleRL-reason/verl/utils/reward_score/simplelr_math.py implements two types of reward functions:

1.Mix Module
```
if correct:
    box_match = 1.0
else:
    box_match = -0.5
if not is_boxed_matched:
    box_match = format_penalty_value  # 默认-1.0
```

2.Independent Module
```
if correct and is_boxed_matched:
    box_match = 1.0
elif correct and not is_boxed_matched:
    box_match = 0.5
elif not correct and is_boxed_matched:
    box_match = -0.5
else:
    box_match = format_penalty_value
```

This project did not explicitly set the REWARD_FUNCTION_TYPE, so it defaulted to the mix mode.


However, for a math reasoning task, the Independent mode is more appropriate.

Therefore, the solution was to modify the training script and pass these variables within the runtime-env-json argument of the ray job submit command.

```
# 格式奖励配置
export REWORD_FUNCTION_TYPE=independent  # 或 mix
export FORMAT_PENALTY_VALUE=-0.3

"env_vars": {
  "http_proxy": "",
  "https_proxy": "",
  "CUDA_VISIBLE_DEVICES": "6,7",
  "REWORD_FUNCTION_TYPE": "'"$REWORD_FUNCTION_TYPE"'",
  "FORMAT_PENALTY_VALUE": "'"$FORMAT_PENALTY_VALUE"'"
}
```

### Experimental Parameters - V2

```
# 格式奖励配置

export REWORD_FUNCTION_TYPE=independent  # 或 mix
export FORMAT_PENALTY_VALUE=-0.3
export PROJECT_NAME=verl_train_2
export WANDB_API_KEY=e193fe755d575bb893b5d1a3034351927c66e91f
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HDFS_DATA_PATH=/simpleRL-reason/data/SimpleRL-Zoo-Data/
export HDFS_MODEL_PATH=/model
export HDFS_CHECKPOINT_PATH=/simpleRL-reason/checkpoints
export HDFS_LOG_PATH=/simpleRL-reason/logs
export RUN_NAME=verl-grpo
export ARNOLD_WORKER_NUM=1 # number of nodes you want to use

  
# 配置Ray对象存储
export RAY_object_store_memory=80000000000  # 80GB（增大了response_length）
export RAY_memory_monitor_refresh_ms=1000    # 1秒检查一次
export RAY_memory_usage_threshold=0.80       # 90%才触发清理
  

# Default values
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=128
MAX_PROMPT_LENGTH=256
MAX_RESPONSE_LENGTH=3840 # The maximum context length for the Qwen2.5-Math-1.5B model is 4096.
LEARNING_RATE=2e-6
PPO_MINI_BATCH_SIZE=128
# per GPU
PPO_MICRO_BATCH_SIZE=2
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=64
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=9  # 大概是500step
DATASET_NAME=simplelr_qwen_level3to5
ROLLOUT_GPU_MEMORY_UTIL=0.6
MODEL_NAME=Qwen2.5-Math-1.5B
SAVE_FREQ=20
TEST_FREQ=99999 # ban
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=512
REMOVE_PREVIOUS_CKPT=False
```

但是结合我之前在log看到的信息，我发现自己遗漏了一个关键点：
```
# MAX_PROMPT_LENGTH = 512
[36m(main_task pid=2341477)[0m original dataset len: 8523
[36m(main_task pid=2341477)[0m filter dataset len: 7119
[36m(main_task pid=2341477)[0m original dataset len: 500
[36m(main_task pid=2341477)[0m filter dataset len: 434
[36m(main_task pid=2341477)[0m Size of train dataloader: 55
[36m(main_task pid=2341477)[0m Size of val dataloader: 1
[36m(main_task pid=2341477)[0m Total training steps: 110

# MAX_PROMPT_LENGTH=256
[36m(main_task pid=3535339)[0m original dataset len: 8523
[36m(main_task pid=3535339)[0m filter dataset len: 1051
[36m(main_task pid=3535339)[0m original dataset len: 500
[36m(main_task pid=3535339)[0m filter dataset len: 95
[36m(main_task pid=3535339)[0m Size of train dataloader: 4
[36m(main_task pid=3535339)[0m Size of val dataloader: 1
[36m(main_task pid=3535339)[0m Total training steps: 20
```

When loading the configuration, the code in /simpleRL-reason/verl/utils/dataset/rl_dataset.py does this:
```
self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
    doc[prompt_key][0]['content']) <= self.max_prompt_length,
                                                     axis=1)]
```

This filtering operation keeps only the data where the prompt content length is $\le$ MAX_PROMPT_LENGTH. This explains why in every previous training run, I saw a prompt_length clip_ratio of 0. The data was being filtered out entirely before training even began.


At 512，Total training steps = Total epochs x Batches per epoch = 2 x \[7119/128] = 2 x 55 = 110

at 256，Total training steps = 5 x \[1051/128] = 5 x 8 = 40


However, the Qwen2.5-Math-1.5B maximum context length is 4096.

```
assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length
# 即：4096 >= 4352  → False → AssertionError
```
Theoretically, prompt_length + response_length must be $\le 4096$.

**The solution is to modify the model's config.json file.**
```
 "max_position_embeddings": 6000,
```

## Validation Parameters

```
bash eval_math_nodes.sh \
    --run_name verl-grpo_simplelr_qwen_level3to5_Qwen2.5-Math-1.5B   \
    --init_model Qwen2.5-Math-1.5B \
    --template qwen-boxed  \
    --tp_size 2 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math \
    --n_sampling 1 \
    --gpus "6,7"
```

## Experiment Results

