from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import NoteNERDataset, NoteNEREvaluator, note_ner_postprocess



prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Extract entity information from the Passage according to the Question requirements. 
Note that the extracted answers should maintain the same case as in the passage.
Here are some examples:

Passage:212 In stock Japanese DIOR 2023 Fall new single color flame gold blush 100/625/343
Question:Please find all 'season' entities
Answer:Fall

Passage:BUTOO tattoo stickers English series colored tattoo stickers waterproof female long-lasting niche high-end tattoo stickers
Question:Please find all 'style' entities
Answer: high-end, niche

Passage:mmm collection slim fit t-shirt women's versatile solid color curved hem short sleeve v-neck fitted hot girl top
Question:Please find all 'style (for clothing and accessories)' entities
Answer: short sleeve, fitted, curved, v-neck

Passage:Early autumn 2022 new sweater plus-size slimming outfit high-end top skirt autumn winter short skirt two-piece set
Question: Please find all 'category' entities
Answer: skirt, sweater, short skirt



Please output the answer (in the same format as the examples above) DIRECTLY after 'Answer:', WITHOUT any additional content.

""")
    ],
    round=[
        dict(role="HUMAN",
             prompt="Passage:{content}\nQuestion:{question}\n"),
        dict(role="BOT", 
             prompt="Answer:", generate=True)
    ]
)


sns_ner_datasets = [
    dict(
        type=NoteNERDataset,
        path="SNS-Bench/sns_bench",
        name='note_ner',
        abbr='sns-NoteNER',
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
            evaluator=dict(type=NoteNEREvaluator),
            pred_postprocessor=dict(
                type=note_ner_postprocess,
                delimiter=",",
            ),
            dataset_postprocessor=dict(
                type=note_ner_postprocess,
                delimiter=",",
            ),
        ),
        reader_cfg=dict(
            input_columns=["question", "content"],
            output_column="answer",
        ),
    )
]
