from mmengine.config import read_base

with read_base():
    from .sns_note_chlw_gen import sns_note_chlw_datasets
    from .sns_note_gender_gen import sns_note_gender_datasets
    from .sns_note_hashtag_gen import sns_note_hashtag_datasets
    from .sns_note_mrc_complex_gen import sns_mrc_c_datasets
    from .sns_note_mrc_simple_gen import sns_mrc_s_datasets
    from .sns_note_ner_gen import sns_ner_datasets
    from .sns_note_querycorr_explain_gen import sns_note_querycorr_explain_datasets
    from .sns_note_querycorr_topic_gen import sns_note_querycorr_topic_datasets
    from .sns_note_querygen_gen import sns_note_querygen_datasets
    from .sns_note_taxonomy_gen import sns_note_taxonomy_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
