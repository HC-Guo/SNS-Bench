from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import NoteCommentHLWDataset, NoteCommentHLWEvaluator


prompt_template=dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Please select words worth highlighting from comments. These words should be meaningful and specific entities, issues, or descriptions that spark users' search interest. 
For life trivia without meaning or search value and content containing numbers, please select carefully. Here are extraction examples:

1) Input: Give me a transparent Bluetooth speaker link; Return: ['transparent Bluetooth speaker']
2) Input: Is there a tutorial for Huawei Smart Screen; Return: ['Huawei Smart Screen tutorial']
3) Input: Is there any specific fitness and weight loss APK?; Return: ['fitness and weight loss APK']
4) Input: Noodles have no calories but the sauce does; Return: ['noodles have no calories but sauce does']
5) Input: How's Peking duck; Return: ['Peking duck']\n\n""")
    ],
    round=[
        dict(role="HUMAN",
             prompt=f"{{content}}",
        ),
        dict(role="BOT", 
             prompt=f"{{answer}}", generate=True)
    ]
)

sns_note_chlw_datasets = [
    dict(
        type=NoteCommentHLWDataset,
        path="SNS-Bench/sns_bench",
        name="note_chlw",
        abbr="sns-NoteCHLW",
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=prompt_template
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=GenInferencer, 
                max_out_len=4096,
                extra_gen_kwargs=dict(
                    temperature=0.7,
                    top_p=0.9,
                ),
            ),
        ),
        eval_cfg=dict(
            evaluator=dict(type=NoteCommentHLWEvaluator),
        ),
        reader_cfg=dict(
            input_columns=["content"],
            output_column="answer",
        ),
    )
]
