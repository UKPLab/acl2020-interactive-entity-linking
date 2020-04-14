from collections import namedtuple
import os
import string
from typing import Dict, List, Optional, Set

import attr

import rdflib

from SPARQLWrapper import SPARQLWrapper, JSON
from  SPARQLWrapper.SPARQLExceptions import EndPointInternalError

from diskcache import Cache

from gleipnir.config import PATH_WWO_PERSONOGRAPHY_RDF, CACHE, PATH_DEPOSITIONS_KB


KbHandle = namedtuple("KbHandle", ["label", "description", "uri"])
NIL = KbHandle("", "", "NIL")


class KnowledgeBase:

    def __init__(self, caching: bool = True):
        self._iri_cache = Cache(os.path.join(CACHE, self.get_name() + "_iri")) if caching else {}
        self._exact_mention_cache = Cache(os.path.join(CACHE, self.get_name() + "_exact")) if caching else {}
        self._fuzzy_mention_cache = Cache(os.path.join(CACHE, self.get_name() + "_fuzzy")) if caching else {}
        self._query_cache = Cache(os.path.join(CACHE, self.get_name() + "_query")) if caching else {}

    def get_by_iri(self, iri: str) -> Optional[KbHandle]:
        cache = self._iri_cache

        if iri == "NIL":
            return KbHandle("", "", "NIL")

        if iri not in cache:
            sparql = SPARQLWrapper(self._get_remote_uri())
            query = self._get_iri_query()
            q = query.replace("$IRI", iri)

            sparql.setQuery(q)
            sparql.setReturnFormat(JSON)

            result = sparql.query().convert()

            handle = self._parse_single(result["results"]["bindings"][0])

            cache[iri] = handle

        return cache[iri]

    def search_mention(self, mention: str) -> Set[KbHandle]:
        exact_cache = self._exact_mention_cache
        fuzzy_cache = self._fuzzy_mention_cache

        if mention not in exact_cache:
            m = self._sanitize_for_fts(mention)
            result_exact_raw = self._execute_search_query(self._get_query_exact_match(), m)
            results_exact = self._parse_result(result_exact_raw)
            exact_cache[mention] = results_exact
        else:
            results_exact = exact_cache[mention]

        if mention not in fuzzy_cache:
            m = self._sanitize_for_fts(mention)
            # m = self._preprocess_mention_fuzzy_search(m)
            result_fuzzy_raw = self._execute_search_query(self._get_query_fuzzy_match(), m)
            results_fuzzy = self._parse_result(result_fuzzy_raw)
            fuzzy_cache[mention] = results_fuzzy
        else:
            results_fuzzy = fuzzy_cache[mention]


        results = results_exact | results_fuzzy
        return results

    def search_user_query(self, mention: str) -> Set[KbHandle]:
        query_cache = self._query_cache
        results = []

        if mention not in query_cache:
            sparql = SPARQLWrapper(self._get_remote_uri())
            query = self._get_query_user_query()

            user_query = self._sanitize_for_fts(mention)
            q = query.replace("$Query", user_query)
            q = q.replace("$query", user_query.lower())

            sparql.setQuery(q)
            sparql.setReturnFormat(JSON)

            try:
                results = self._parse_result(sparql.query().convert())
            except EndPointInternalError as e:
                print(e)

            query_cache[mention] = results
        else:
            results = query_cache[mention]

        return results

    def _get_iri_query(self):
        raise NotImplementedError()

    def _get_query_exact_match(self) -> str:
        """ The string should contain `$Mention` as a placeholder for the upper case mention
        and `$mention` for the lower case mention.
        """
        raise NotImplementedError()

    def _get_query_fuzzy_match(self) -> str:
        raise NotImplementedError()

    def _get_query_user_query(self) -> str:
        raise NotImplementedError()

    def _get_remote_uri(self) -> str:
        raise NotImplementedError()

    def _execute_search_query(self, query: str, mention: str) -> List[KbHandle]:
        sparql = SPARQLWrapper(self._get_remote_uri())

        q = query.replace("$Mention", mention)
        q = q.replace("$mention", mention.lower())
        q = q.replace('"Mention"', f'"{mention}"')

        sparql.setQuery(q)
        sparql.setReturnFormat(JSON)

        try:
            return sparql.query().convert()
        except EndPointInternalError as e:
            if "XM028" in e.args[0]:
                return []
            raise e

    def _sanitize_for_fts(self, mention: str) -> str:
        s = mention.strip().translate(str.maketrans('', '', string.punctuation))
        return s.strip()

    def _preprocess_mention_fuzzy_search(self, mention: str) -> str:
        return mention

    def _parse_result(self, results) -> Set[KbHandle]:
        handles = set()
        if len(results) == 0:
            return handles

        for e in results["results"]["bindings"]:
            handle = self._parse_single(e)
            handles.add(handle)

        return handles

    def _parse_single(self, e: Dict[str, str]) -> KbHandle:
        label = e["lc"]["value"] if "lc" in e else ""
        description = e["dc"]["value"] if "dc" in e else ""
        iri = e["subj"]["value"]
        handle = KbHandle(label=label, description=description, uri=iri)
        return handle

    def get_name(self) -> str:
        pass


class WikidataVirtuosoKnowledgeBase(KnowledgeBase):

    def _get_iri_query(self):
        return """SELECT DISTINCT ?dc ?lc ?subj WHERE {
        BIND( <$IRI> AS ?subj) 
        OPTIONAL
          {   { ?subj  <http://schema.org/description>  ?dc
                FILTER langMatches(lang(?dc), "en")
              }
            UNION
              { ?subj  <http://schema.org/description>  ?dc
                FILTER langMatches(lang(?dc), "")
              }
          }
    
        OPTIONAL
          {   { ?subj rdfs:label  ?lc
                FILTER langMatches(lang(?lc), "")
              }
            UNION
              { ?subj  rdfs:label  ?lc
                FILTER langMatches(lang(?lc), "en")
              }
          }
      
    }
    LIMIT 2"""

    def _get_query_exact_match(self) -> str:
        return """SELECT DISTINCT ?lc ?dc ?subj ?pLabel
        WHERE { { { { VALUES ( ?pLabel ) { (<http://www.w3.org/2004/02/skos/core#prefLabel>) (<http://www.w3.org/2004/02/skos/core#altLabel>) }
        { ?subj ?pLabel ?lc .
        ?lc <bif:contains> '''"Mention"''' .
        FILTER ( LCASE( STR( ?lc ) ) = "$mention"@en || LCASE( STR( ?lc ) ) = "$mention" ) } }
        FILTER EXISTS { ?subj <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> [ ] .
        FILTER NOT EXISTS { ?subj <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2004/02/skos/core#Concept> . }
        FILTER NOT EXISTS { [ ] <http://www.w3.org/2004/02/skos/core#broader> ?subj . }
        FILTER NOT EXISTS { ?subj <http://www.w3.org/2004/02/skos/core#broader> [ ] . } } } }
        OPTIONAL { { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } } }
        LIMIT 10"""

    def _get_query_fuzzy_match(self) -> str:
        return """SELECT DISTINCT ?lc ?dc ?subj ?pLabel
        WHERE { { { { VALUES ( ?pLabel ) { (<http://www.w3.org/2004/02/skos/core#prefLabel>) (<http://www.w3.org/2004/02/skos/core#altLabel>) }
        { ?subj ?pLabel ?lc .
        ?lc <bif:contains> '''"Mention"''' .
        FILTER ( ( CONTAINS( LCASE( STR( ?lc ) ), "$mention" ) && LANGMATCHES( LANG( ?lc ), "en" ) ) || ( CONTAINS( LCASE( STR( ?lc ) ), "$mention" ) && LANGMATCHES( LANG( ?lc ), "" ) ) ) } }
        FILTER EXISTS { ?subj <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> [ ] .
        FILTER NOT EXISTS { ?subj <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2004/02/skos/core#Concept> . }
        FILTER NOT EXISTS { [ ] <http://www.w3.org/2004/02/skos/core#broader> ?subj . }
        FILTER NOT EXISTS { ?subj <http://www.w3.org/2004/02/skos/core#broader> [ ] . } } } }
        OPTIONAL { { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } } }
        LIMIT 10"""

    def _get_query_user_query(self) -> str:
        return """SELECT DISTINCT ?lc ?dc ?subj
        WHERE { { { { OPTIONAL { ?pLabel <http://www.wikidata.org/prop/direct/P1647>* <http://www.w3.org/2000/01/rdf-schema#label> . }
        { ?subj ?pLabel ?lc .
        ?lc <bif:contains> '''"$query*"''' .
        FILTER ( LANGMATCHES( LANG( ?lc ), "en" ) || LANGMATCHES( LANG( ?lc ), "" )) } }
        FILTER EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> [ ] .
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q35120> . }
        FILTER NOT EXISTS { [ ] <http://www.wikidata.org/prop/direct/P279> ?subj . }
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P279> [ ] . } } } }
        OPTIONAL { { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } } }
        LIMIT 2000"""

    def _get_remote_uri(self) -> str:
        return "http://knowledgebase.ukp.informatik.tu-darmstadt.de:8890/sparql"

    def get_name(self) -> str:
        return "wikidata-virtuoso"


class FusekiKnowledgeBase(KnowledgeBase):

    def __init__(self, name: str, caching: bool = True):
        self._name = name
        super().__init__(caching=caching)

    def _get_iri_query(self):
        return """SELECT DISTINCT ?dc ?lc ?subj
        WHERE { { { VALUES ( ?subj ) { (<$IRI>) }  } }
        OPTIONAL { { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } }
        OPTIONAL { ?pLabel <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>* <http://www.w3.org/2004/02/skos/core#prefLabel> . }
        OPTIONAL { { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "" ) ) } UNION { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "en" ) ) } } }
        LIMIT 2"""

    def _get_query_exact_match(self) -> str:
        return """PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?dc ?lc ?subj
        WHERE {
          { ?subj skos:prefLabel "$mention" . }
            UNION
          { ?subj skos:prefLabel "$Mention" . }
          ?subj skos:prefLabel ?lc .
          ?subj rdfs:comment ?dc
        }
        LIMIT 5"""

    def _get_query_fuzzy_match(self) -> str:
        return """
            PREFIX  text: <http://jena.apache.org/text#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT  ?lc ?dc ?subj
            WHERE
              { 
                  ?subj text:query ( skos:prefLabel '$mention' ) .
                  ?subj skos:prefLabel ?lc .
                  ?subj <http://www.w3.org/2000/01/rdf-schema#comment>  ?dc
              }
            LIMIT   2000"""

    def _get_query_user_query(self) -> str:
        return """
            PREFIX  text: <http://jena.apache.org/text#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT  ?lc ?dc ?subj
            WHERE
              { 
                  ?subj  text:query ( skos:prefLabel "$query*" ) ; skos:prefLabel ?lc .
                      
                OPTIONAL
                  {   { ?subj  <http://www.w3.org/2000/01/rdf-schema#comment>  ?dc
                        FILTER langMatches(lang(?dc), "en")
                      }
                    UNION
                      { ?subj  <http://www.w3.org/2000/01/rdf-schema#comment>  ?dc
                        FILTER langMatches(lang(?dc), "")
                      }
                  }
              }
            LIMIT   2000"""

    def _preprocess_mention_fuzzy_search(self, mention: str) -> str:
        return " ".join([s + "~" if len(s) > 3 else s for s in mention.split()])

    def _get_remote_uri(self) -> str:
        return f"http://localhost:3030/{self._name}/query"

    def get_name(self) -> str:
        return self._name


class WikidataKnowledgeBase(KnowledgeBase):

    def _get_iri_query(self) -> str:
        return """SELECT DISTINCT ?dc ?lc ?subj
        WHERE { { { VALUES ( ?subj ) { (<$IRI>) }  } }
        OPTIONAL { { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } }
        OPTIONAL { ?pLabel <http://www.wikidata.org/prop/direct/P1647>* <http://www.w3.org/2000/01/rdf-schema#label> . }
        OPTIONAL { { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "" ) ) } UNION { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "en" ) ) } } }
        LIMIT 2"""

    def _get_query_exact_match(self) -> str:
        return """SELECT DISTINCT ?lc ?dc ?subj
        WHERE { { { { OPTIONAL { ?pLabel <http://www.wikidata.org/prop/direct/P1647>* <http://www.w3.org/2000/01/rdf-schema#label> . }
        { SERVICE wikibase:mwapi { 
        bd:serviceParam wikibase:api "EntitySearch" . 
        bd:serviceParam wikibase:endpoint "www.wikidata.org" . 
        bd:serviceParam wikibase:limit 50 . 
        bd:serviceParam mwapi:search "Mention" . 
        bd:serviceParam mwapi:language "en" . 
        ?subj wikibase:apiOutputItem mwapi:item . 
        }
        { ?subj ?pLabel ?lc .
        FILTER ( LCASE( STR( ?lc ) ) = "$mention"@en || LCASE( STR( ?lc ) ) = "$mention" ) } } }
        FILTER EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> [ ] .
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q35120> . }
        FILTER NOT EXISTS { [ ] <http://www.wikidata.org/prop/direct/P279> ?subj . }
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P279> [ ] . } } } }
        OPTIONAL { { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } } }
        LIMIT 2000"""

    def _get_query_fuzzy_match(self) -> str:
        return """SELECT DISTINCT ?lc ?dc ?subj
        WHERE { { { { OPTIONAL { ?pLabel <http://www.wikidata.org/prop/direct/P1647>* <http://www.w3.org/2000/01/rdf-schema#label> . }
        { SERVICE wikibase:mwapi { 
        bd:serviceParam wikibase:api "EntitySearch" . 
        bd:serviceParam wikibase:endpoint "www.wikidata.org" . 
        bd:serviceParam wikibase:limit 50 . 
        bd:serviceParam mwapi:search "Mention" . 
        bd:serviceParam mwapi:language "en" . 
        ?subj wikibase:apiOutputItem mwapi:item . 
        }
        { ?subj ?pLabel ?lc .
        FILTER ( ( CONTAINS( LCASE( STR( ?lc ) ), "$mention" ) && LANGMATCHES( LANG( ?lc ), "en" ) ) || ( CONTAINS( LCASE( STR( ?lc ) ), "$mention" ) && LANGMATCHES( LANG( ?lc ), "" ) ) ) } } }
        FILTER EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> [ ] .
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q35120> . }
        FILTER NOT EXISTS { [ ] <http://www.wikidata.org/prop/direct/P279> ?subj . }
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P279> [ ] . } } } }
        OPTIONAL { { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } } }
        LIMIT 2000
        """

    def _get_query_user_query(self) -> str:
        return """SELECT DISTINCT ?lc ?dc ?subj
        WHERE { { { { OPTIONAL { ?pLabel <http://www.wikidata.org/prop/direct/P1647>* <http://www.w3.org/2000/01/rdf-schema#label> . }
        { SERVICE wikibase:mwapi { 
        bd:serviceParam wikibase:api "EntitySearch" . 
        bd:serviceParam wikibase:endpoint "www.wikidata.org" . 
        bd:serviceParam wikibase:limit "once" . 
        bd:serviceParam mwapi:search "$query" . 
        bd:serviceParam mwapi:language "en" . 
        ?subj wikibase:apiOutputItem mwapi:item . 
        }
        { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "en" ) ) } } }
        FILTER EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> [ ] .
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q35120> . }
        FILTER NOT EXISTS { [ ] <http://www.wikidata.org/prop/direct/P279> ?subj . }
        FILTER NOT EXISTS { ?subj <http://www.wikidata.org/prop/direct/P279> [ ] . } } } }
        OPTIONAL { { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://schema.org/description> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } } }
        LIMIT 2000       
        """

    def _get_remote_uri(self) -> str:
        return "https://query.wikidata.org/sparql"

    def get_name(self) -> str:
        return "wikidata"


class LocalKnowledgeBase:

    def __init__(self, name: str):
        if name == "1641":
            path = PATH_DEPOSITIONS_KB
        else:
            raise Exception(f"Unknown kb name: [{name}]")

        g = rdflib.Graph()
        g.parse(path, format="ttl")

        self._graph = g

    def get_by_iri(self, iri: str) -> Optional[KbHandle]:
        query = """SELECT DISTINCT ?dc ?lc ?subj
        WHERE { { { VALUES ( ?subj ) { (<$IRI>) }  } }
        OPTIONAL { { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "en" ) ) } UNION { ?subj <http://www.w3.org/2000/01/rdf-schema#comment> ?dc .
        FILTER ( LANGMATCHES( LANG( ?dc ), "" ) ) } }
        OPTIONAL { ?pLabel <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>* <http://www.w3.org/2004/02/skos/core#prefLabel> . }
        OPTIONAL { { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "" ) ) } UNION { ?subj ?pLabel ?lc .
        FILTER ( LANGMATCHES( LANG( ?lc ), "en" ) ) } } }
        LIMIT 2"""

        q = query.replace("$IRI", iri)
        qres = self._graph.query(q)

        for row in qres:
            print(row)

        return qres

    def search(self, mention: str) -> Set[KbHandle]:
        pass


def load_wwo_kb() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(PATH_WWO_PERSONOGRAPHY_RDF, format="ttl")
    return g


def compute_kb_confusability_wwo(g: rdflib.Graph, surface_form: str) -> int:
    query = """SELECT (COUNT(distinct ?s) AS ?candidates) 
        WHERE { 
            { ?s skos:prefLabel ?surface_form }
            UNION
            { ?s skos:altLabel  ?surface_form }
        }
    """

    qres = g.query(query, initBindings={"surface_form": rdflib.Literal(surface_form)})
    for row in qres:
        return int(row[0])





