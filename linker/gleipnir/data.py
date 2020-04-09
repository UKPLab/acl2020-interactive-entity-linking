from collections import defaultdict
from typing import List, Set

import attr

from gleipnir.corpora import *
from gleipnir.kb import WikidataVirtuosoKnowledgeBase, KbHandle, NIL, KnowledgeBase, FusekiKnowledgeBase


class CandidateGenerator:

    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    def generate_negative_train_candidates(self, gold_iri: str, mention: str) -> List[KbHandle]:
        """ Removes gold entity """
        candidates = self._get_candidates(mention)
        candidates = {c for c in candidates if c.uri != gold_iri}
        candidates.add(NIL)
        return list(candidates)

    def get_gold_entity(self, gold_iri: str) -> KbHandle:
        return self._kb.get_by_iri(gold_iri)

    def generate_test_candidates(self, mention: str) -> List[KbHandle]:
        candidates = self._get_candidates(mention)
        if len(candidates) == 0:
            candidates.add(NIL)
        return list(candidates)

    def generate_simulation_candidates_phase1(self, mention: str) -> List[KbHandle]:
        pass

    def generate_simulation_candidates_phase2(self, mention: str) -> List[KbHandle]:
        pass

    def generate_simulation_candidates_phase3(self, mention: str) -> List[KbHandle]:
        pass

    def _get_candidates(self, mention) -> Set[KbHandle]:
        kb_candidates = self._kb.search_mention(mention)
        return kb_candidates


