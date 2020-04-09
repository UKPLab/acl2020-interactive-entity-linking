from typing import List, Tuple

from gleipnir.config import *
from gleipnir.converter.wikipedia import map_wikipedia_url_to_wikidata_url
from gleipnir.formats.conllel import *


def _load_raw_aida() -> Tuple[Corpus, Corpus, Corpus]:
    train = []
    dev = []
    test = []

    tokens = []
    entities = {}
    token_idx = 0
    last_entity_begin = -1

    def _parse_document_header(document_header: str) -> List[str]:
        if "testa " in document_header:
            return dev
        elif "testb " in document_header:
            return test
        else:
            return train

    with open(PATH_AIDA_RAW) as f:
        for line in f:
            line = line.strip()

            if line.startswith("-DOCSTART-"):
                name = line[12:-1]
                document = Document(name)
                target = _parse_document_header(line)
                target.append(document)
            elif not line:
                # End of sentence

                sentence = Sentence(tokens, entities)
                document.sentences.append(sentence)

                tokens = []
                entities = {}
                token_idx = 0
                last_entity_begin = -1
            else:
                # Parse line
                e = line.split("\t")

                tokens.append(Token(token_idx, e[0]))

                if len(e) == 1:
                    pass
                elif len(e) == 4 or len(e) == 6 or len(e) == 7:
                    # Parse entity annotation

                    if e[1] != "B":
                        # If it is a continuing annotation, we just increase the span of the
                        # already existing annotation
                        entity = entities[last_entity_begin]
                        entity.end += 1
                    else:
                        assert len(e) != 4 or e[3] == "--NME--"
                        label = "NIL" if len(e) == 4 else e[4]

                        entity = Entity(start=token_idx, end=token_idx + 1, label=label)
                        entities[token_idx] = entity
                        last_entity_begin = token_idx
                elif len(e) == 4:
                    pass
                else:
                    raise RuntimeError(f"Invalid line: {line}")

                token_idx += 1

    return Corpus(train), Corpus(dev), Corpus(test)


def convert_aida():
    def _convert_single(data: Corpus, path: str):
        for document in data.documents:
            for sentence in document.sentences:
                for entity in sentence.entities.values():
                    entity.uri = map_wikipedia_url_to_wikidata_url(entity.uri)

        write_conllel(path, data)

    # AIDA comes with one file that has train, dev, test all in it,
    # we split it first and then save them separately
    train, dev, test = _load_raw_aida()
    _convert_single(train, PATH_AIDA_WIKIDATA_TRAIN)
    _convert_single(dev, PATH_AIDA_WIKIDATA_DEV)
    _convert_single(test, PATH_AIDA_WIKIDATA_TEST)


if __name__ == '__main__':
    convert_aida()
