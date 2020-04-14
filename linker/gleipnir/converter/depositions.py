import json
import urllib
from collections import defaultdict
import csv
from typing import List, Dict

from SPARQLWrapper import SPARQLWrapper, JSON

import rdflib
from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import ClosedNamespace, RDF, RDFS

import attr
from tqdm import tqdm

from gleipnir.config import *
from gleipnir.corpora import load_depositions_test_raw, load_depositions_dev_raw, load_depositions_train_raw, load_depositions_raw_all
from gleipnir.formats import Corpus, write_conllel
from gleipnir.handcrafted import FeatureGenerator


@attr.s
class Person:
    old_iri = attr.ib()             # type: str
    new_iri = attr.ib()             # type: str
    label = attr.ib()               # type: str
    alt_label = attr.ib()           # type: str
    description = attr.ib()         # type: str
    birth_place = attr.ib()         # type: str
    title = attr.ib()               # type: str


@attr.s
class Place:
    old_iri = attr.ib()             # type: str
    new_iri = attr.ib()             # type: str
    label = attr.ib()               # type: str
    alt_label = attr.ib()           # type: str
    description = attr.ib()         # type: str


def convert_persons_from_ods():
    remap = {}
    persons = []

    with open(PATH_DEPOSITIONS_PERSON_KB_RAW, newline='') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"', )

        # Skip header
        next(reader)

        for e in reader:
            old_iri, is_duplicate, label, alt_label, description, birth_place, title, _ = e

            if old_iri.startswith("http://dbpedia.org/"):
                dbpedia_label, description = get_dpbedia_label_description(old_iri)
                if not label:
                    label = dbpedia_label

            new_iri = build_new_iri(label)
            remap[old_iri] = new_iri

            if is_duplicate == "x":
                continue

            person = Person(old_iri, new_iri, label.strip(), alt_label.strip(), description.strip(), birth_place.strip(), title.strip())
            persons.append(person)

    return persons, remap


def convert_places_from_ods():
    remap = {}
    places = []

    with open(PATH_DEPOSITIONS_PLACES_KB_RAW, newline='') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"', )

        # Skip header
        next(reader)

        for e in reader:
            old_iri, is_duplicate, label, alt_label, description = e
            label = label.strip()

            if old_iri.startswith("http://dbpedia.org/"):
                dbpedia_label, description = get_dpbedia_label_description(old_iri)
                if not label.strip():
                    label = dbpedia_label

            new_iri = build_new_iri(label)

            remap[old_iri] = new_iri

            if is_duplicate == "x":
                continue

            place = Place(old_iri, new_iri, label.strip(), alt_label.strip(), description.strip())
            places.append(place)

    return places, remap


def get_dpbedia_label_description(iri: str):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")

    query = """select distinct ?label ?description where {
    <$iri> rdfs:label ?label .
    <$iri> rdfs:comment	 ?description .
    FILTER (LANG(?label)='en' && LANG(?description) = 'en') 
    } LIMIT 1""".replace("$iri", iri)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        result = sparql.query().convert()
        bindings = result["results"]["bindings"][0]

        return bindings["label"]["value"], bindings["description"]["value"]
    except:
        return "", ""


def depositions_to_rdf(persons: List[Person], places: List[Place], entity_to_documents: Dict[str, str]):
    g = rdflib.Graph()

    skos_terms = [
        "Collection", "Concept", "ConceptScheme", "OrderedCollection", "altLabel", "broadMatch",
        "broader", "changeNote", "closeMatch", "definition", "editorialNote", "exactMatch",
        "example", "hasTopConcept", "hiddenLabel", "historyNote", "inScheme", "mappingRelation",
        "member", "memberList", "narrowMatch", "narrower", "narrowerTransitive", "notation", "note",
        "prefLabel", "related", "relatedMatch", "scopeNote", "semanticRelation", "topConceptOf"
    ]

    skos = ClosedNamespace(
        uri=URIRef("http://www.w3.org/2004/02/skos/core#"),
        terms=skos_terms
    )
    depositions = Namespace("http://www.gleipnir.de/1641/ns/1.0#")

    g.bind('skos', skos)
    g.bind('depo', depositions)

    depo_person = depositions.Person
    depo_has_title = depositions.hasTitle
    depo_place_of_birth = depositions.placeOfBirth
    depo_mentioned_in = depositions.mentionedIn

    depo_place = depositions.Place
    depo_county = depositions.County
    depo_in_county = depositions.county

    depo_dbpedia = depositions.dbpedia

    properties = [
        depo_person,
        depo_place,
        depo_has_title,
        depo_place_of_birth,
        depo_dbpedia,
        depo_mentioned_in,
        depo_county,
        depo_in_county
    ]

    for prop in properties:
        g.add((prop, RDF.type, RDF.Property))

    g.add((depo_person, skos.broader, skos.Concept))
    g.add((depo_place, skos.broader, skos.Concept))
    g.add((depo_county, skos.broader, skos.Concept))

    for person in persons:
        s = URIRef(person.new_iri)

        g.add((s, RDF.type, depo_person))

        if person.label:
            g.add((s, skos.prefLabel, Literal(person.label)))

        if person.description:
            g.add((s, RDFS.comment, Literal(person.description)))

        if person.title:
            g.add((s, depo_has_title, Literal(person.title)))

        if person.alt_label:
            g.add((s, skos.altLabel, Literal(person.alt_label)))

        if person.birth_place:
            g.add((s, depo_place_of_birth, Literal(person.birth_place)))

        if person.old_iri.startswith("http://dbpedia.org/"):
            g.add((s, depo_dbpedia, URIRef(person.old_iri)))

        for doc in entity_to_documents[person.old_iri]:
            g.add((s, depo_mentioned_in, URIRef("http:" + doc)))

    counties = [
        "Antrim",
        "Armagh",
        "Carlow",
        "Cavan",
        "Clare",
        "Cork",
        "Derry",
        "Donegal",
        "Down",
        "Dublin",
        "Fermanagh",
        "Galway",
        "Kerry",
        "Kildare",
        "Kilkenny",
        "Laois",
        "Leitrim",
        "Limerick",
        "Longford",
        "Louth",
        "Mayo",
        "Meath",
        "Monaghan",
        "Offaly",
        "Roscommon",
        "Sligo",
        "Tipperary",
        "Tyrone",
        "Waterford",
        "Westmeath",
        "Wexford",
        "Wicklow",
    ]

    # Convert counties
    for county in counties:
        county_iri = f"http://www.gleipnir.de/1641/ns/1.0#County_{county}"
        s = URIRef(county_iri)

        g.add((s, RDF.type, depo_county))

        dbpedia_iri = f"http://dbpedia.org/resource/County_{county}"

        g.add((s, depo_dbpedia, URIRef(dbpedia_iri)))
        label, description = get_dpbedia_label_description(dbpedia_iri)

        if label:
            g.add((s, skos.prefLabel, Literal(label)))

        if description:
            g.add((s, RDFS.comment, Literal(description)))

    # Convert places
    for place in places:
        s = URIRef(place.new_iri)

        g.add((s, RDF.type, depo_place))

        if place.label:
            g.add((s, skos.prefLabel, Literal(place.label)))

        if place.description:
            g.add((s, RDFS.comment, Literal(place.description)))

        if place.alt_label:
            g.add((s, skos.altLabel, Literal(place.alt_label)))

        if place.old_iri.startswith("http://dbpedia.org/"):
            g.add((s, depo_dbpedia, URIRef(place.old_iri)))

        if "County" in place.description and not place.old_iri.startswith("http://dbpedia.org/"):
            name = place.description.split()[-1]
            county_iri = build_new_iri(f"County_{name}")
            g.add((s, depo_in_county, URIRef(county_iri)))

        for doc in entity_to_documents[place.old_iri]:
            g.add((s, depo_mentioned_in, URIRef("http:" + doc)))

    os.makedirs(os.path.dirname(PATH_DEPOSITIONS_KB), exist_ok=True)

    with open(PATH_DEPOSITIONS_KB, "wb") as f:
        f.write(g.serialize(format='turtle', encoding="utf-8"))


def load_all_irish_people() -> List[Person]:
    result = []
    with open(PATH_DEPOSITIONS_PERSON_JSON) as f:
        data = json.load(f)

        for o in data["results"]["bindings"]:
            iri = o["subj"]["value"]
            label = o["label"]["value"]
            description = o["desc"]["value"]

            new_iri = build_new_iri(label)
            person = Person(iri, new_iri, label.strip(), None, description.strip(), None, None)
            result.append(person)

    return result


def load_all_irish_places() -> List[Place]:
    result = []
    with open(PATH_DEPOSITIONS_PLACES_JSON) as f:
        data = json.load(f)

        for o in data["results"]["bindings"]:
            iri = o["subj"]["value"]
            label = o["label"]["value"]
            description = o["desc"]["value"]

            new_iri = build_new_iri(label)
            place = Place(iri, new_iri, label.strip(), None, description.strip())
            result.append(place)

    return result


def remap_corpus(corpus: Corpus, remap: Dict[str, str], target_path: str):
    for document in corpus.documents:
        for sentence in document.sentences:
            for entity in sentence.entities.values():
                entity.uri = remap[entity.uri]

    write_conllel(target_path, corpus)


def build_new_iri(label: str):
    cleaned = urllib.parse.quote('_'.join(label.split()), safe='')
    assert cleaned
    new_iri = f"http://www.gleipnir.de/1641/ns/1.0#{cleaned}"
    return new_iri


def main():
    corpus = load_depositions_raw_all()

    entity_to_documents = defaultdict(list)
    for document in corpus.documents:
        for sentence in document.sentences:
            for entity in sentence.entities.values():
                entity_to_documents[entity.uri].append(document.name)

    persons, remap_persons = convert_persons_from_ods()
    places, remap_places = convert_places_from_ods()

    print(f"Persons ods: {len(persons)}")
    print(f"Places ods: {len(places)}")

    print(f"Persons added: {len(load_all_irish_people())}")
    print(f"Places added: {len(load_all_irish_places())}")

    persons.extend(load_all_irish_people())
    places.extend(load_all_irish_places())

    remap = {**remap_persons, **remap_places}

    depositions_to_rdf(persons, places, entity_to_documents)

    preload_cache = False
    if preload_cache:
        fg = FeatureGenerator()

        for person in tqdm(persons):
            fg._sentencebert_cd("", "context", "label", person.description)

        for place in tqdm(places):
            fg._sentencebert_cd("", "context", "label", place.description)

    # Remap stuff
    remap_corpus(load_depositions_train_raw(), remap, PATH_DEPOSITIONS_TRAIN_REMAPPED)
    remap_corpus(load_depositions_dev_raw(), remap, PATH_DEPOSITIONS_DEV_REMAPPED)
    remap_corpus(load_depositions_test_raw(), remap, PATH_DEPOSITIONS_TEST_REMAPPED)


if __name__ == '__main__':
    main()
