import torch
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFacewithChatTemplate
from mmengine.config import read_base


"""
run command:
    opencompass examples/eval_sns_bench.py -w outputs/sns_bench
"""


with read_base():
    from opencompass.configs.datasets.sns_bench.sns_bench_gen import datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-7B-Instruct',
        path='/mnt/nlp-ali/usr/checkpoints/opensource/Qwen2.5-7B-Instruct',
        tokenizer_path='/mnt/nlp-ali/usr/checkpoints/opensource/Qwen2.5-7B-Instruct',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        generation_kwargs=dict(
            do_sample=True,
            temperature=0.1,
            top_p=0.01,
        ),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=32,
        batch_padding=False,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=10000,    # max size in each task
        gen_task_coef=40,       # expansioin ratio for generative task
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        max_workers_per_gpu=2,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask),
    ),
)