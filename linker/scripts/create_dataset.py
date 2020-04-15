import os

from gleipnir.config import PATH_HANDCRAFTED
from gleipnir.datasets import get_raw_corpus_data
from gleipnir.handcrafted import Feature, FeatureGenerator, WikidataVirtuosoKnowledgeBase
from gleipnir.kb import FusekiKnowledgeBase, KnowledgeBase


def get_kb(kb_name: str, cached: bool = True) -> KnowledgeBase:
    if kb_name == "wwo-fuseki":
        return FusekiKnowledgeBase("wwo", caching=cached)
    elif kb_name == "1641-fuseki":
        return FusekiKnowledgeBase("depositions", caching=cached)
    elif kb_name == "aida":
        return WikidataVirtuosoKnowledgeBase(caching=cached)
    else:
        raise ValueError(f"Invalid model name: {kb_name}")


def main():
    os.makedirs(PATH_HANDCRAFTED, exist_ok=True)

    names = [
        # "aida",
        "wwo-fuseki",
        # "1641-fuseki",
    ]

    for name in names:
        data = get_raw_corpus_data(name)
        kb = get_kb(name, cached=True)
        features = list(Feature)
        # features = [Feature.LENGTH_CONTEXT]
        fg = FeatureGenerator()

        df_train = fg.compute_features_for_corpus(name + "_train", data.corpus_train, data.cg, features, True)
        df_dev = fg.compute_features_for_corpus(name + "_dev", data.corpus_dev, data.cg, features, True)
        df_test = fg.compute_features_for_corpus(name + "_test", data.corpus_test, data.cg, features, True)
        df_all = fg.compute_features_for_corpus(name + "_all", data.corpus_all, data.cg, features, True)

        df_train.ext.to_csv()
        df_dev.ext.to_csv()
        df_test.ext.to_csv()
        df_all.ext.to_csv()

        df_sim = fg.compute_features_for_simulation(name + "_full_sim", data.corpus_all, kb, features)
        df_sim.ext.to_csv()


if __name__ == '__main__':
    main()