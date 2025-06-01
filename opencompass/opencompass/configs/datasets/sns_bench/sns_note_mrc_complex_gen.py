from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import NoteMRCComplexDataset, NoteMRCComplexEvaluator



prompt_template=dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""Task: First rewrite the Query, then perform reading comprehension.

###Reference Content Format###
* The reference content consists of three parts: Query, Doc, and Level
* Query is the user's search term with question-asking intent; Doc is the main text part of the image note, wrapped in <Doc> and </Doc>; Level defines the expected answer type, indicating what type of answer the user needs.

###Task Requirements###
Step 1: Based on Query and Level, rewrite the Query into specific answer requirements, please output the rewritten rewrite_query. Here are some examples for the first step:
1. [Query=World's most beautiful museums recommendations, Level=museum names]=>[Names of the world's most beautiful museums]
2. [Query=Coody Coffee recommendations, Level=drink names]=>[Delicious drink names at Coody Coffee]
3. [Query=Qingdao food, Level=dishes]=>[Delicious dishes in Qingdao]
4. [Query=Games similar to Number Bomb, Level=game names]=>[What games are similar to Number Bomb]
5. [Query=Birthday cake recommendations in Licheng District, Jinan, Level=bakery cake names]=>[Delicious birthday cakes from bakeries in Licheng District, Jinan]
6. [Query=Guangzhou Yuexiu District food recommendations, Level=restaurant dishes]=>[Delicious restaurant dishes in Yuexiu District, Guangzhou]
7. [Query=Hiking survival bracelet recommendations, Level=brand model]=>[Brand and model recommendations for hiking survival bracelets]
8. [Query=Must-buy in Japan, Level=brand model]=>[Brand and model of Japanese specialty products]
9. [Query=iPhone 11 photography recommendations, Level=model]=>[iPhone 11 models good for photography]
10. [Query=1000 desk essentials, Level=product names]=>[1000 useful products for desks]
11. [Query=Changsha special breakfast recommendations, Level=restaurants]=>[Names of restaurants with special breakfast in Changsha]
12. [Query=Nanjing surrounding tours, Level=cities]=>[Cities suitable for tourism around Nanjing]
13. [Query=Cheap clothing places in Shenzhen, Level=shopping locations]=>[Names of shopping locations for cheap clothes in Shenzhen]
14. [Query=Harbin bathhouse recommendations, Level=bathhouse names]=>[Names of worthwhile bathhouses in Harbin]

Step 2: Evaluate the relevance between Doc and rewrite_query, paying attention to important qualifiers in rewrite_query, such as time and location modifiers. If Doc misses important qualifiers, it can be considered irrelevant. Analyze the relevance in 1-2 sentences.

Step 3: If Doc is relevant to rewrite_query, extract all answers (Options) from Doc that can answer the user's question and put them in AnswerList. Options must match the answer type required by rewrite_query. Specifically, if rewrite_query requires "brand model", ensure each answer Option includes both "brand name" and "model"; if this cannot be satisfied, consider it unextractable. If rewrite_query requires "restaurant dishes", ensure each answer Option includes both "restaurant name" and "dish name". Others like "bakery cake names" follow the same pattern. If Option cannot meet the required answer type, consider it unextractable, and AnswerList will be an empty list [].

Step 4: For each answer Option in AnswerList, extract a related description (Reason) from Doc. Format: [{"Option": "xxx", "Reason": "xxx"}, ...]. If Doc is irrelevant or has no answers, return an empty list [].

Notes:
1. Answer Options must follow the original text, non-continuous extraction is allowed.
2. Reasons must be extracted from the original text, no modifications allowed.
3. Answer Options should typically be specific entity nouns or names that can be directly searched through search engines; avoid extracting vague descriptions like "this hidden gem hotel" or "rice noodles from the shop near XXX bus station".
4. Doc may have various article structures, including a common parallel point structure with leading words. When extracting answer Options, avoid including leading words like "Figure 1", "Figure 2", ..., "Shop 1", "Shop 2", etc.
5. Reason length should not exceed 100 english character widths.


Output format:
<rewrite_query>...</rewrite_query>
<relevance_analysis>...</relevance_analysis>
<AnswerList>[\"...\", ...]</AnswerList>
<Result>[{\"Option\": \"xxx\", \"Reason\": \"xxx\"}, ...]</Result>\n\n\n

""")
    ],
    round=[
        dict(role="HUMAN",
             prompt="{content}"),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)

sns_mrc_c_datasets = [
    dict(
        type=NoteMRCComplexDataset,
        path="SNS-Bench/sns_bench",
        name="note_mrc_complex",
        abbr='sns-NoteMRC-C',
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
            evaluator=dict(type=NoteMRCComplexEvaluator),
        ),
        reader_cfg=dict(
            input_columns=["content"],
            output_column="answer",
        ),
    )
]




