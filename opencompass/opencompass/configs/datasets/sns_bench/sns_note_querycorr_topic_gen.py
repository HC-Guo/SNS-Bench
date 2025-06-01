from opencompass.openicl import ZeroRetriever, PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sns_bench import NoteQueryCorrTopicDataset, NoteQueryCorrTopicEvaluator



prompt_template = dict(
    begin=[
        dict(role="SYSTEM", fallback_role="HUMAN", 
             prompt="""You are a relevance scoring robot. You will receive a query and a Xiaohongshu note content, and you will score the relevance between the note and the query. 
Before scoring, I will first introduce you to the concepts of requirements and topics, and then provide you with rating levels to choose from.

First, I will introduce you to the concepts of requirements and topics.

Requirements:
```
In search engines, users express their search intent through queries. However, due to factors such as knowledge level and cognition, the input queries may not accurately describe users' true needs.
Therefore, search systems need to accurately understand the real needs behind queries to retrieve the content users need.
Generally, queries can be divided into "Precise Need Queries" and "General Need Queries" based on requirements. Their definitions and differences are as follows:
* Precise Need Queries: queries with clear and unique intentions.
e.g. how to treat knee pain, how to cook tomato and eggs
* General Need Queries: queries that may have multiple intentions, which can be further divided into two categories according to primary and secondary needs:
** Primary Needs: The most direct and basic expectations when users search, covering common user needs. Usually what users first think of and most want to get answers to immediately.
** Secondary Needs: Some additional needs around the primary needs, or specific needs. Although these needs aren't common to all users, they are very important to some users.
```

Topics:
```
Can be understood as core issues or involved entity objects, and notes' topics can be viewed from different angles, examples as follows:
Example 1:
618 must-haves‼️Sharing happiness-boosting bathroom fixtures😌
Abstract perspective: Home bathroom essentials
Specific perspective: Introduction of bathroom cabinet, smart toilet, shower head

Example 2:
Five easy-to-raise dogs that don't smell
Abstract perspective: Introduction to easy-to-raise dogs
Specific perspective: Introduction of 5 dogs: Schnauzer, Poodle, Bichon Frise, Shiba Inu, Pomeranian

Based on the matching degree between "query topic" and "note topic", they can be divided into 3 categories:
* Category 1: Complete topic match: The note's topic matches the query topic
For example:
Query=home bathroom recommendations, Note=618 must-haves‼️Sharing happiness-boosting bathroom fixtures😌
Query=which dogs are easy to raise, Note=Five easy-to-raise dogs that don't smell

* Category 2: Partial topic match: Part of the note's topic matches the query topic
For example:
Query=smart toilet, Note=618 must-haves‼️Sharing happiness-boosting bathroom fixtures😌
Query=poodle, Note=Five easy-to-raise dogs that don't smell

* Category 3: No topic match: The note's topic doesn't match the query topic at all
For example:
Query=ceiling decoration recommendations, Note=618 must-haves‼️Sharing happiness-boosting bathroom fixtures😌
Query=are kangaroos easy to raise, Note=Five easy-to-raise dogs that don't smell
```


Available rating levels:
```
* 3: Meets primary needs, complete topic match (relevant content ≥80%)
Examples:
Query=basketball Note=Who invented basketball?

* 2: Meets primary needs, partial topic match (relevant content between 10%~80%)
Examples:
Query=Schnauzer vs Poodle comparison Note=Five easy-to-raise dogs that don't smell  | [Meets secondary needs, complete topic match (relevant content ≥80%)]
Query=Apple Note=After becoming famous, Fan Bingbing's most regrettable movie "Apple" | [Meets secondary needs, partial topic match (relevant content between 10%~80%)]
Query=Xiaomi su7 Note=Help! Should I choose su7 or Mercedes c260

* 1: Low satisfaction level, relevant content less than 10%
Examples:
Query=23-week glucose tolerance Note=Peking Union Medical College Hospital International Birth Record - Prenatal Care | [Low satisfaction level, only mentioned]
Query=Can gold be purchased Note=Today's gold price-May 18, 2024 <Up> | [Low satisfaction level, extremely specific need]
Query=How to play badminton Note=About sharing a racket with my bestie to play badminton | [Low satisfaction level, helpful for query with matching topic]
Query=Quick weight loss exercise Note=100 reps daily, anytime anywhere💦belly fat + thigh fat reduction❗️effective

* 0: Doesn't meet needs, has some connection, keywords match
Examples:
Query=Gemini wants to break up Note=How long does it take for Gemini to forget ex  [Doesn't meet needs, has some connection, no keyword match]
Query=Chaotianmen Hotpot Note=Haidilao please stop posting on social media

* -1: Doesn't meet needs, no connection at all
Examples:
Query=Amazing girls Note=These three makeup schools are not recommended, ordinary girls can't afford

```

""")
    ],
    round=[
        dict(role='HUMAN',
             prompt="""The Query and note you received are:
{content}
Please output the rating level, choosing from the five numerical levels: -1, 0, 1, 2, 3. 
Output only the numerical rating, without any additional content.

"""),
        dict(role="BOT",
             prompt="{answer}", generate=True)
    ]
)


sns_note_querycorr_topic_datasets = [
    dict(
        type=NoteQueryCorrTopicDataset,
        path="SNS-Bench/sns_bench",
        name="note_querycorr_topic",
        abbr='sns-NoteQueryCorr-topic',
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
            evaluator=dict(type=NoteQueryCorrTopicEvaluator),
        ),
        reader_cfg=dict(
            input_columns=["content"],
            output_column="answer",
        ),
    )
]
