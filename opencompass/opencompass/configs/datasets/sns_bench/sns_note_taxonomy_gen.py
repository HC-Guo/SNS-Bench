from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, EMEvaluator
from opencompass.datasets.sns_bench import NoteTaxonomyDataset, note_taxonomy_postprocess


# single
prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Given a SNS note's title and content, and a list of candidate categories, please select the most suitable category from the candidate category list. 
Here are some examples:

Input: Note title: Need help, is this a car dash cam\n Note content: Bought a DV for 250, came with a battery and charging cable, and it's very light, wondering if it's really a dash cam[crying R][crying R]#DV[topic]# 
Candidate categories: Car knowledge, Trending, Car lifestyle, Car accessories, Motorcycles, Car shopping, Car modification, New energy & smart, Car culture, Driving test learning, Other automotive
Answer: Car accessories

Input: Note content: Traveling Henan · Understanding China | Xinxiang South Taihang Tourism Resort: Let's follow the lens to see the sea of clouds in South Taihang #TravelingHenanUnderstandingChina#HomelandHenan#HomelandHenanNewMediaMatrix#May19ChinaTourismDay#XinxiangSouthTaihangTourismResort#SeaOfClouds#XinxiangCultureBroadcastingForeignAffairsAndTourismBureau 
Candidate categories: Travel VLOG, Shopping, Food exploration, Places to go, Travel guide, Travel records, Hotels & B&Bs, Attraction experience, Travel tips, Living abroad, Indoor leisure 
Answer: Travel records



Please output the answer (in the same format as the examples above) DIRECTLY after 'Answer:', WITHOUT any additional content.

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Input: {content}\nCandidate categories: {candidates}\nAnswer: "),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_taxonomy_single_datasets = [
    dict(
        type=NoteTaxonomyDataset,
        path="SNS-Bench/sns_bench",
        name="note_taxonomy_single",
        abbr="sns-NoteTaxonomy-single",
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
            pred_postprocessor=dict(
                type=note_taxonomy_postprocess,
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
             prompt="""Given a SNS note's title and content, and lists of candidate primary, secondary, and tertiary categories, please select the most suitable categories from each level. Answer in the format: Primary category | Secondary category | Tertiary category. 
Here are some examples:

Input: Note title: The super disappointing AHC facial cleanser for me\n Note content: During National Day trip to Hong Kong, several colleagues asked me to help buy AHC facial cleanser, saying it was especially good.\nI followed the trend and bought one, used it in the hotel that night💥💢After washing, my eyes were particularly painful, even a tiny bit made me feel like I was going blind👉🙅🙅so where exactly is it good😤😤😱😱Now a full bottle is sitting at home, won't repurchase.\nBought for about 70 HKD ~ equivalent to about 60 RMB. The Innisfree green tea cleanser is still better for an affordable cleanser😖😖😖 
Candidate primary categories: ['Music', 'Beauty', 'Mother&Baby', 'Outdoor', 'Humanities', 'Photography', 'Gaming', 'Art', 'Trends', 'Entertainment', 'Film&TV', 'Kids', 'Career', 'Food']
Candidate secondary categories: ['Education Daily', 'Skincare', 'Other Pets', 'Handicraft', 'Resource Sharing', 'Exhibition', 'Personal Care', 'Shoes', 'Mobile Games', 'Relationship Knowledge', 'Meals', 'Places', 'Language Education', 'Makeup']
Candidate tertiary categories: ['Cycling Records', 'Playlist Sharing', 'Cleansing', 'Other Shoes', 'Fruit', 'Self-study', 'Instrument Playing', 'JK', 'Beverage Review', 'Tablet', 'Career Development', 'Boutique', 'Other Campus', 'Driving Safety']
Answer: Beauty | Skincare | Cleansing

Input: Note title: Day8\n Note content: Forgot to post yesterday\nMaking up for it today#ChineseBrushCalligraphyCheckIn[topic]# #DailyCalligraphyCheckIn[topic]# # 
Candidate primary categories: ['Anime', 'Beauty', 'Gaming', 'Humanities', 'Home&Decoration', 'Health', 'Relationships', 'Social Science', 'Career', 'Art', 'Education', 'Life Records']
Candidate secondary categories: ['Science', 'Car Lifestyle', 'Running', 'Other', 'Culture', 'Weight Loss Medicine', 'Parks', 'Fashion', 'Accessories', 'Weight Loss', 'Music Sharing', 'Finger Gaming']
Candidate tertiary categories: ['Bags', 'Fruit', 'Leisure Guide', 'Sheet Music Sharing', 'Font Design', 'Calligraphy', 'Weight Loss Tutorial', 'Snack Review', 'Life Science', 'Swimwear', 'Ball Sports', 'Skincare Collection']
Answer: Humanities | Culture | Calligraphy



Please output the answer (in the same format as the examples above) DIRECTLY after 'Answer:', WITHOUT any additional content.

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="Input: {content}\nCandidate primary categories: {candidates_primary}\nCandidate secondary categories: {candidates_secondary}\nCandidate tertiary categories: {candidates_tertiary}\nAnswer: "),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)

sns_note_taxonomy_multi_datasets = [
    dict(
        type=NoteTaxonomyDataset,
        path="SNS-Bench/sns_bench",
        name="note_taxonomy_multiple",
        abbr="sns-NoteTaxonomy-multi",
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
            evaluator=dict(type=EMEvaluator),
            pred_postprocessor=dict(
                type=note_taxonomy_postprocess,
            ),
        ),
        reader_cfg=dict(
            input_columns=["content", "candidates_primary", "candidates_secondary", "candidates_tertiary"],
            output_column="answer",
        ),
    )
]


sns_note_taxonomy_datasets  = sns_note_taxonomy_single_datasets + sns_note_taxonomy_multi_datasets
datasets = sns_note_taxonomy_multi_datasets