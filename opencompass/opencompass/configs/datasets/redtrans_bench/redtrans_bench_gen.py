from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.redtrans_bench import RedTransDataset, RedTransEvaluator


system_prompt = """
You are a professional, authentic translation engine. specializing in translation from {source_lang} to {target_lang}.
You only return the translated text, without any explanations.
""".strip()
user_prompt = """
DEFINE ROLE AS "Xiaohongshu Linguistic Translator":
  task = "This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text."
  expertise = ["social_media_slang", "cultural_localization", "internet_memes", "emoji_conversion"]

FUNCTION TRANSLATE(source_text):
  constraint = "Do not provide any explanations or text apart from the translation."
  quality = ["complete", "accurate", "faithful", "natural", "fluent", "readable", "culturally adapted"]
  free_translation = DO_FREE_TRANSLATION(source_text, quality, constraint)

  RETURN free_translation

MAIN PROCESS:
  source_text = INPUT(\"""{user_input}\""")
  translated_text = TRANSLATE(source_text)

  RETURN translated_text

Only return the translated_text, ensuring it contains no explanations and no {source_lang} characters.
""".strip()


redtrans_tasks = {
    "en2zh"   : {
        "system_prompt" : system_prompt.format(source_lang="English", target_lang="Chinese"),
        "user_prompt"   : user_prompt.format(source_lang="English", target_lang="Chinese", user_input="{en}"),
    },
    "zh2en"   : {
        "system_prompt" : system_prompt.format(source_lang="Chinese", target_lang="English"),
        "user_prompt"   : user_prompt.format(source_lang="Chinese", target_lang="English", user_input="{zh}"),
    }
}


redtrans_datasets = []
for task_name in redtrans_tasks.keys():
    redtrans_datasets.append(
        dict(
            type=RedTransDataset,
            path='RedTransBench/redtrans_bench',
            name=f'redtrans_{task_name}',
            abbr=f"redtrans-{task_name}",
            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        begin=[
                            dict(role="SYSTEM", fallback_role="HUMAN",
                                prompt=redtrans_tasks[task_name]["system_prompt"],
                            )
                        ],
                        round=[
                            dict(role="HUMAN",
                                prompt=redtrans_tasks[task_name]["user_prompt"],
                            ),
                            dict(role="BOT", 
                                prompt="{label}", generate=True)
                        ]
                    )
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=4096),
            ),
            eval_cfg=dict(
                evaluator=dict(type=RedTransEvaluator),
            ),
            reader_cfg=dict(
                input_columns=[task_name.split('2')[0]],
                output_column="label",
            ),
        )
    )

