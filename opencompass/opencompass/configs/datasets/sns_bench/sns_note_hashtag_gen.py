from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import (
    NoteHashtagDataset, NoteHashtagNEREvaluator, note_hashtag_single_postprocess, note_hashtag_multi_postprocess
)



# single
prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Give you a SNS note title and content, along with a list of candidate topic tags, please select the most relevant tag from the list for the note. 
Below are some examples:

Input: 
Note Title: ❤️Enfj Big Swords Will Love Life BGM Recommendations 🔥\n Note Content: Compiled some BGMs for everyone, if you have any recommended songs, please leave them in the comments section, let's exchange and share together 😀
Candidate Topic Tags: 
National Peace and Tranquility, Outfit for Pear-Shaped Body, High EQ Small Tricks, Vienna Guide
Answer: High EQ Small Tricks

Input: 
Note Title: Must-Have 🩱 at the Sanya Beach\n Note Content: On a summer beach, how can you do without this blue tie-dye dress! 💙\t\n\t\nSize XL, perfectly fits plus-size sisters! 👯‍♀️\t\n\t\nA high-end, laid-back style, loose design, wear it and instantly look whiter and slimmer! 🩱\t\n\t\nThis dress is not only fashionable but also of great quality, a must-have item for beach vacations! 🏖️\t\n\t\nEven the most picky of you will fall in love with it! 💖
Candidate Topic Tags: 
National Trend Short Sleeves, Beach Outfit, Miracle Warm Pictures, Badminton Uniform Outfit
Answer: Beach Outfit



Please output the answer (in the same format as the examples above) DIRECTLY after 'Answer:', WITHOUT any additional content.

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Input: \n{content}\nCandidate Topic Tags: \n{candidates}\nAnswer: "),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_hashtag_single_datasets = [
    dict(
        type=NoteHashtagDataset,
        path="SNS-Bench/sns_bench",
        name="note_hashtag_single",
        abbr="sns-NoteHashtag-single",
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=prompt_template,
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
            pred_postprocessor=dict(
                type=note_hashtag_single_postprocess,
            ),
        ),
        reader_cfg=dict(
            input_columns=["content", "candidates"],
            output_column="answer",
        ),
    )
]




# multi
prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Given a title and content of a SNS note, and a list of candidate topic tags, please select all the tags that match the note from the list of topic tags.
Here are some examples:

Input: 
Note Title: Deposit Money in Wuhan\n Note Content: Squat for a Wuhan customer manager \n Deposit for 5 years!!!
Candidate Topic Tags: 
September Birthday, DIY in Guangzhou, Savings Check-in, Wuhan, Henna Tattoo, Nap Time, Vancouver Mercedes, Must-Have Outdoor Gear, Bank, Liu Xiang, Savings, 5D Stereoscopic Mask, Sam's Club, Non-Cyan Earring Recommendations
Answer: Wuhan, Savings, Savings Check-in, Bank

Input: 
Note Title: First Time Making Handmade Balls, Afraid of Sealing Issues!!\n Note Content: Love handmade balls so much! Never thought making them myself would be so much fun! But the glue for sealing is really hard to handle. Use too little and it might not be sturdy, use too much and it might burn [Cry R][Cry R]. How long should I wait before peeling off the film? 🐷Girls interested can check my page, switched to a new account, will be sharing more textured balls\n52💼u (5 balls)
Candidate Topic Tags: 
CPA Exam Review, After Being a Mom, Recruiting Agents, Manicure Wholesale Collaboration, First Time Making Handmade Balls, Korea Trip, Shanghai Yuanxing Huayu Real Estate Group Co., Ltd., Handmade Ball, Pinching Ball Fun, Popular Mask Recommendations, Resume Coaching, Handmade Ball FX, Handmade Ball Sharing, New Driving Style, Pinching Sharing, Basic Skills Competition for Class Teachers, Handmade Ball Original
Answer: Pinching Sharing, Handmade Ball Sharing, Handmade Ball FX, Handmade Ball Original, First Time Making Handmade Balls, Handmade Ball, Pinching Ball Fun



Please output the answer (in the same format as the examples above) DIRECTLY after 'Answer:', WITHOUT any additional content.

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Input: \n{content}\nCandidate Topic Tags: \n{candidates}\nAnswer: "),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_hashtag_multi_datasets =[
    dict(
        type=NoteHashtagDataset,
        path="SNS-Bench/sns_bench",
        name="note_hashtag_multiple",
        abbr="sns-NoteHashtag-multi",
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
            evaluator=dict(type=NoteHashtagNEREvaluator),
            pred_postprocessor=dict(
                type=note_hashtag_multi_postprocess,
            ),
            dataset_postprocessor=dict(
                type=note_hashtag_multi_postprocess,
            ),
        ),
        reader_cfg=dict(
            input_columns=["content", "candidates"],
            output_column="answer",
        ),
    )
]


sns_note_hashtag_datasets = sns_note_hashtag_single_datasets + sns_note_hashtag_multi_datasets

