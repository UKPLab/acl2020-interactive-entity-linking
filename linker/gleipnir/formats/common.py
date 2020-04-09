from typing import Dict, List, Iterator, Optional

import attr


@attr.s
class Corpus:
    documents = attr.ib(factory=list)   # type: List[Document]
    name = attr.ib(default="")        # type: str

    def iter_entities(self) -> Iterator["Entity"]:
        for document in self.documents:
            for sentence in document.sentences:
                yield from sentence.entities.values()

    def number_of_documents(self) -> int:
        return len(self.documents)

    def number_of_tokens(self) -> int:
        count = 0
        for document in self.documents:
            for sentence in document.sentences:
                count += len(sentence.tokens)
        return count

    def number_of_entities(self) -> int:
        count = 0
        for document in self.documents:
            for sentence in document.sentences:
                count += len(sentence.entities)
        return count

    def entities_per_sentence(self) -> float:
        number_of_sentences = 0
        number_of_entities = 0

        for document in self.documents:
            for sentence in document.sentences:
                number_of_sentences += 1
                number_of_entities += len(sentence.entities)

        return number_of_entities / number_of_sentences


@attr.s
class Document:
    name = attr.ib()                    # type: str
    sentences = attr.ib(factory=list)   # type: List[Sentence]


@attr.s
class Sentence:
    tokens = attr.ib()      # type: List[Token]
    entities = attr.ib()    # type: Dict[int, Entity]
    idx = attr.ib(default=None)         # type: Optional[int]

    def get_covered_text(self, entity: "Entity") -> str:
        return " ".join(t.text for t in self.tokens[entity.start:entity.end])

    def get_text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    def __str__(self):
        " ".join([x.text for x in self.tokens]) + "; " + " ".join([x.uri for x in self.entities.values()])


@attr.s(frozen=True)
class Token:
    idx = attr.ib()         # type: int
    text = attr.ib()        # type: str


@attr.s
class Entity:
    start = attr.ib()       # type: int
    end = attr.ib()         # type: int
    uri = attr.ib()       # type: str


@attr.s
class DataPoint:
    mention = attr.ib()                 # type: str
    context_lst = attr.ib()             # type: List[str]
    uri = attr.ib()                     # type: str

    @property
    def context(self) -> str:
        return " ".join(self.context_lst)

    def lower(self) -> 'DataPoint':
        return DataPoint(self.mention.lower(), [x.lower() for x in self.context_lst], self.uri)


def convert_to_dataset(corpus: Corpus, context_size = 2) -> List[DataPoint]:
    result = []

    for document in corpus.documents:
        for sentence in document.sentences:
            for entity in sentence.entities.values():
                surface_form = sentence.get_covered_text(entity)


                idx = sentence.idx
                context = document.sentences[
                          max(0, idx - context_size): min(len(document.sentences), idx + context_size + 1)]
                context = [c.get_text() for c in context]

                point = DataPoint(surface_form, context, entity.uri)
                result.append(point)

    return result
