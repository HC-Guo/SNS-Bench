from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.sns_bench import NoteGenderDataset, note_gender_postprocess


prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""According to the following note content fields, determine whether the content is likely to trigger discussions of interest to both men and women. 
If it includes such content, reply 'Yes,' otherwise reply 'No.' Only reply with 'Yes' or 'No,' without any additional explanation.

These contents mainly include the following aspects: 
Stereotypes about men and women;
Gender bias phenomena, gender inequality in occupations, sports, interests, etc.;
Fertility-related content: including views on childbirth, parenting experiences, etc.;
Marriage-related content: including matchmaking, weddings, post-marriage life, mother-in-law relationships, marital breakdown, prenuptial property, etc.;
Romance-related content: emotional experiences, arguments, scumbag men/women, cheating, etc.; 
Sex-related content: including sexual violence, sexual harassment, discussions on sexual knowledge, etc.;
And some related social events.
If the text mentions these topics, it is likely to trigger discussions of interest to both men and women.

Here are some examples: 
Note Content:
Title: When I deliberately sleep in separate beds from my boyfriend to see his reaction\nCategory: Emotions-Daily Life\nText: In the end it really scared me #Love[Topic]# #Boyfriend[Topic]# #Couple Daily[Topic]# \nasr:When I deliberately sleep in separate beds from my boyfriend...(rest of content)
Answer:Yes

Note Content:
Title:《The Young Lady's Meng》\nCategory: Humanities-Reading\nText:#Baby Food[Topic]##Accessories Share[Topic]# \nasr:\nocr:...(rest of content)
Answer:No

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Please provide your judgment on the following content: \nNote Content:\n{content}\nAnswer:"),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_gender_datasets = [
    dict(
        type=NoteGenderDataset,
        path="SNS-Bench/sns_bench",
        name="note_gender",
        abbr='sns-NoteGender',
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=prompt_template
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=GenInferencer, 
                max_out_len=2048,
                extra_gen_kwargs=dict(
                    temperature=0.7,
                    top_p=0.9,
                ),
            ),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=note_gender_postprocess)
        ),
        reader_cfg=dict(
            input_columns=["content"],
            output_column="answer",
        ),
    )
]
