from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.sns_bench import NoteQueryCorrExplainDataset, note_querycorr_explain_postprocess

prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Below is a question and a document. Please determine if the given document can answer the given question. 
If it can, reply with "Yes", otherwise reply with "No". Only reply with "Yes" or "No", no additional explanations are needed. 
Here are some examples:

Question: 
Is the distorted face in the back camera of an iPhone the real you? 
Document: 
Revealed | Is the you in the original camera really you? |- Distorted features? Facial asymmetry? Poor skin condition? Skewed face? Why do you see all these when you open the original camera on your phone? And then you get anxious: Am I really that ugly? Actually, it's not the case. The original camera on your phone does indeed make you look worse. Let me explain one by one about appearance anxiety #notcamerafriendly
Answer: No

Question: 
Is the corn used in cultural playthings real corn? 
Document: 
Light Macaroon | Beautiful, beautiful | All real corn grown, nature's colorful gifts currently popular cultural plaything corn (naturally grown) undyed
Answer: Yes

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Question: \n{sentence1}\nDocument: \n{sentence2}\nAnswer: "),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_querycorr_explain_datasets = [
    dict(
        type=NoteQueryCorrExplainDataset,
        path="SNS-Bench/sns_bench",
        name="note_querycorr_explain",
        abbr='sns-NoteQueryCorr-explain',
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
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=note_querycorr_explain_postprocess)
        ),
        reader_cfg=dict(
            input_columns=["sentence1", "sentence2"],
            output_column="answer",
        ),
    )
]
