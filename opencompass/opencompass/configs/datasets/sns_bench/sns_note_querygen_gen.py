from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import NoteQueryGenDataset, NoteQueryGenEvaluator



prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Assume you are an internet user, your task is: based on the input image/text or video note content, generate 1 novel search term related to the core entity words in the note.

Output format: 
The return result is a single line string, recording the generated search term.
Output requirements:
1. You need to fully understand the input note information, ensure that the given search terms are complete, novel and interesting, with the possibility of being targeted or in-depth exploration, and can bring incremental information to the current note;
2. Ensure that the given search terms are fluent, without expression problems (including incomprehensible, repetitive, typos, word order, etc.);
3. Good search terms can be questions (such as how to xx, what is xx, how to xx etc.), or appropriate compound words (xx tips, xx tutorials, xx outfits, xx recommendations, xx collection, xx selection), where xx comes from the note content.
4. Output search term DIRECTLY, without any additional analysis content.

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Please generate search terms for the following note based on the above instructions: \nInput: \n{content}\nSearch term:\n"),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_querygen_datasets = [
    dict(
        type=NoteQueryGenDataset,
        path="SNS-Bench/sns_bench",
        name="note_querygen",
        abbr='sns-NoteQueryGen',
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
            evaluator=dict(
                type=NoteQueryGenEvaluator,
                embed_model_path="/mnt/nlp-ali/usr/checkpoints/opensource/bge-large-en-v1.5",
            ),
        ),
        reader_cfg=dict(
            input_columns=["content"],
            output_column="answer",
        ),
    )
]
