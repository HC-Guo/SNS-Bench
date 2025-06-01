from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import NoteMRCSimpleDataset, NoteMRCSimpleEvaluator



prompt_template=dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""You are a reading comprehension master: 
I will provide you with a user's note article, including a title and content, where the title may be empty "". I will also give you a user's question. 

Please think according to the following steps:
* Step 1, combine the note's title and content to determine if the user's question can be answered. If there are specific entity words or qualifiers in the question, they must be satisfied to count as having an answer.
* Step 2, if Step 1 determines that the content can answer the user's question, then extract the answer from the original note content and output the extracted original content.

Content requirements for answers:
* Extract sentences that can answer the question, cross-sentence extraction is allowed, but must be complete sentences or paragraphs, multiple consecutive sentences must be extracted continuously, and preserve special characters like \n.
* If the middle part of consecutive sentences is irrelevant to the answer, it can be skipped for cross-sentence extraction; if the answer is long (more than 120 characters), it can be extracted by points.
* Content must come from the original text, no summarization allowed; do not extract content unrelated to answering the user's question.

Format requirements for answers:
* Output the answers in list form, where each value in the list is a sentence or paragraph that can answer the question.
* If no answer can be extracted, return an empty list [].
* Do not output any content other than the answer list.


""")
    ],
    round=[
        dict(role="HUMAN",
             prompt="""Here is the note title and content:\n\n{content}\n\n\nUser's question:\n{query}\nOutput:\n""",),
        dict(role="BOT", 
             prompt="{answer}", generate=True)
    ]
)

sns_mrc_s_datasets = [
    dict(
        type=NoteMRCSimpleDataset,
        path="SNS-Bench/sns_bench",
        name="note_mrc_simple",
        abbr='sns-NoteMRC-S',
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=prompt_template
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=GenInferencer, 
                max_out_len=4096,        # 99% of the answers' token length is less than 400, max is 730, avg is 34.54
                extra_gen_kwargs=dict(
                    temperature=0.7,
                    top_p=0.9,
                ),
            ),
        ),
        eval_cfg=dict(
            evaluator=dict(type=NoteMRCSimpleEvaluator),
        ),
        reader_cfg=dict(
            input_columns=["query", "content"],
            output_column="answer",
        ),
    )
]
