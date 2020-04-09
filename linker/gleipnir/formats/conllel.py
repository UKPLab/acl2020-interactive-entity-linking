import logging
import os

from gleipnir.formats import *

_logger = logging.getLogger(__file__)


def read_conllel(file_name: str, with_labels=True) -> Corpus:
    _logger.debug("Read connel file from [%s]", file_name)

    documents = []
    tokens = []
    entities = {}
    token_idx = 0
    sentence_idx = 0

    with open(file_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("-DOCSTART-"):
                sentence_idx = 0
                header = line[line.find("("):-1]
                parts = header.split()
                name = parts[1] if len(parts) > 1 else parts[0]

                document = Document(name)
                documents.append(document)
            elif not line:
                document.sentences.append(Sentence(tokens, entities, sentence_idx))
                sentence_idx += 1

                tokens = []
                entities = {}
                token_idx = 0
            else:
                e = line.split("\t")

                tokens.append(Token(token_idx, e[0]))

                if len(e) not in (1, 3):
                    raise RuntimeError(f"Invalid line: {line}")

                if len(e) == 3:
                    if with_labels:
                        label = e[2]
                    else:
                        label = None

                    start, end = [int(x) for x in e[1].split(":")]
                    entities[token_idx] = Entity(start, end, label)

                token_idx += 1

    return Corpus(documents, os.path.basename(file_name))


def write_conllel(file_name: str, corpus: Corpus):
    with open(file_name, "w") as f:
        for document in corpus.documents:
            f.write(f"-DOCSTART- ({document.name})\n")

            for sentence in document.sentences:
                for token in sentence.tokens:
                    f.write(token.text)

                    if token.idx in sentence.entities:
                        entity = sentence.entities[token.idx]
                        f.write("\t")
                        f.write(f"{entity.start}:{entity.end}")
                        f.write("\t")
                        f.write(entity.uri)

                    f.write("\n")

                f.write("\n")

            f.write("\n")


