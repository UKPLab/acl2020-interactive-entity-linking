from datetime import datetime

import os

# Paths

ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_DATA = os.path.join(ROOT, "data")
PATH_GENERATED = os.path.join(ROOT, "generated")
PATH_RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")

PATH_MODELS = os.path.join(PATH_GENERATED, "models")

PATH_PLOTS = os.path.join(PATH_RESULTS, "plots")

PATH_GERNED = os.path.join(PATH_DATA, "gerned")
PATH_GERNED_RAW = os.path.join(PATH_GERNED, "ANY_german_queries_with_answers.xml")
PATH_GERNED_TAB = os.path.join(PATH_GERNED, "ANY_german_queries.tab")
PATH_GERNED_WIKIDATA = os.path.join(PATH_GERNED, "gerned_wikidata.csv")
PATH_GERNED_NEWS = os.path.join(PATH_GERNED, "dataset", "news")
PATH_GERNED_PROCESSED = os.path.join(PATH_GENERATED, "gerned.conll")

PATH_AIDA_RAW = os.path.join(PATH_DATA, "aida-yago2-dataset", "AIDA-YAGO2-dataset.tsv")
PATH_AIDA_WIKIDATA_TRAIN = os.path.join(PATH_GENERATED, "aida-yago2-dataset", "aida_wikidata_train.conll")
PATH_AIDA_WIKIDATA_DEV = os.path.join(PATH_GENERATED, "aida-yago2-dataset", "aida_wikidata_dev.conll")
PATH_AIDA_WIKIDATA_TEST = os.path.join(PATH_GENERATED, "aida-yago2-dataset", "aida_wikidata_test.conll")

PATH_APH_RAW = os.path.join(PATH_DATA, "aph")
PATH_APH_RAW_TRAIN = os.path.join(PATH_APH_RAW, "goldset")
PATH_APH_RAW_DEV = os.path.join(PATH_APH_RAW, "devset")
PATH_APH_RAW_TEST = os.path.join(PATH_APH_RAW, "testset")

PATH_APH = os.path.join(PATH_GENERATED, "aph")
PATH_APH_TRAIN = os.path.join(PATH_APH, "aph_train.conll")
PATH_APH_DEV = os.path.join(PATH_APH, "aph_dev.conll")
PATH_APH_TEST = os.path.join(PATH_APH, "aph_test.conll")

PATH_WWO_PERSONOGRAPHY_RAW = os.path.join(ROOT, "..", "data-converter", "wwo", "personography.xml")
PATH_WWO_PERSONOGRAPHY_RDF = os.path.join(PATH_GENERATED, "wwo", "personography.ttl")
PATH_WWO_DOCUMENTS_RAW = os.path.join(PATH_DATA, "wwo", "files")

PATH_WWO_TRAIN = os.path.join(PATH_GENERATED, "wwo", "train.conll")
PATH_WWO_DEV = os.path.join(PATH_GENERATED, "wwo", "dev.conll")
PATH_WWO_TEST = os.path.join(PATH_GENERATED, "wwo", "test.conll")

PATH_DEPOSITIONS_TRAIN = os.path.join(PATH_GENERATED, "depositions", "train.conll")
PATH_DEPOSITIONS_DEV = os.path.join(PATH_GENERATED, "depositions", "dev.conll")
PATH_DEPOSITIONS_TEST = os.path.join(PATH_GENERATED, "depositions", "test.conll")

PATH_DEPOSITIONS_TRAIN_REMAPPED = os.path.join(PATH_GENERATED, "depositions", "train_remapped.conll")
PATH_DEPOSITIONS_DEV_REMAPPED = os.path.join(PATH_GENERATED, "depositions", "dev_remapped.conll")
PATH_DEPOSITIONS_TEST_REMAPPED = os.path.join(PATH_GENERATED, "depositions", "test_remapped.conll")

PATH_DEPOSITIONS_PERSON_KB_RAW = os.path.join(PATH_DATA, "depositions", "kb", "irish_kb_persons.csv")
PATH_DEPOSITIONS_PLACES_KB_RAW = os.path.join(PATH_DATA, "depositions", "kb", "irish_kb_places.csv")
PATH_DEPOSITIONS_PERSON_JSON = os.path.join(PATH_DATA, "depositions", "kb", "person.json")
PATH_DEPOSITIONS_PLACES_JSON = os.path.join(PATH_DATA, "depositions", "kb", "places.json")
PATH_DEPOSITIONS_KB = os.path.join(PATH_GENERATED, "depositions", "kb", "depositions_kb.ttl")

PATH_WIKIDATA_RAW = os.path.join(PATH_DATA, "wikidata", "latest-all.json.bz2")

PATH_WIKIPEDIA_PAGES_RAW = os.path.join(PATH_DATA, "wikipedia", "enwiki-latest-page.sql.gz")
PATH_WIKIPEDIA_PAGE_PROPS_RAW = os.path.join(PATH_DATA, "wikipedia", "enwiki-latest-page_props.sql.gz")
PATH_WIKIPEDIA_REDIRECTS_RAW = os.path.join(PATH_DATA, "wikipedia", "enwiki-latest-redirect.sql.gz")

PATH_WIKIPEDIA_PAGE_IDS = os.path.join(PATH_GENERATED, "wikipedia", "enwiki-ids.tsv")
PATH_WIKIPEDIA_PAGE_PROPS = os.path.join(PATH_GENERATED, "wikipedia", "enwiki-wikidata-ids.tsv")

PATH_WIKIPEDIA_WIKIDATA_INDEX = os.path.join(PATH_GENERATED, "wikipedia", "enwiki-wikidata-mapping.db")

PATH_HANDCRAFTED = os.path.join(PATH_GENERATED, "handcrafted")

PATH_RECOMMENDER_MODELS = os.path.join(ROOT, "recommender_models")

PATH_DATA_USERSTUDY = os.path.join(PATH_DATA, "userstudy")

def path_to_results() -> str:
    path = os.path.join(PATH_RESULTS, datetime.now().isoformat().replace(":", "-"))
    os.makedirs(path, exist_ok=True)
    return path
