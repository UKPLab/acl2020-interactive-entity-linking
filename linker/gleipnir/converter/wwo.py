from gleipnir.config import *
from itertools import chain
from typing import List

from lxml import etree

import attr

import rdflib
from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import ClosedNamespace, RDF, RDFS


NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "teib": "http://www.wwp.northeastern.edu/ns/textbase"
}


@attr.s
class Person:
    uid = attr.ib()
    label = attr.ib()
    description = attr.ib()
    roles = attr.ib()
    alternative_labels = attr.ib()
    date_of_birth = attr.ib()
    date_of_death = attr.ib()
    place_of_birth = attr.ib()
    place_of_death = attr.ib()
    faith = attr.ib()


def load_personography() -> List[Person]:
    tree = etree.parse(PATH_WWO_PERSONOGRAPHY_RAW)
    root = tree.getroot()

    print(PATH_WWO_PERSONOGRAPHY_RAW)

    text = root.find("tei:text", NS)
    body = text.find("tei:body", NS)
    listPerson = body.find("tei:listPerson", NS)

    persons = []
    for person_xml in listPerson.findall('tei:person', NS):
        # ID parsing

        xml_id = person_xml.get("{http://www.w3.org/XML/1998/namespace}id")

        # Label name parsing
        persName = person_xml.find("tei:persName", NS)
        surname = persName.find("tei:surname", NS)
        forenames = persName.findall("tei:forename", NS)

        name_parts = []

        if forenames:
            name_parts.extend(forenames)

        if surname is not None:
            name_parts.append(surname)

        label = " ".join([x.text for x in name_parts])

        # Role parsing
        roles = [x.text for x in persName.findall("tei:roleName", NS)]

        # Alternative names parsing
        alternative_labels = []

        alternative_labels.extend(x.text for x in persName.findall(".//tei:name/[@type='non-breaking']", NS))
        alternative_labels.extend(x.text for x in person_xml.findall(".//tei:persName/[@type='variant']", NS))

        if not label and alternative_labels:
            label = alternative_labels.pop(0)

        # Date of birth/death

        date_of_birth_raw = person_xml.find(".//tei:birth/[@when]", NS)
        date_of_birth = date_of_birth_raw.get("when") if date_of_birth_raw is not None else None

        date_of_death_raw = person_xml.find(".//tei:death/[@when]", NS)
        date_of_death = date_of_death_raw.get("when") if date_of_death_raw is not None else None

        # Place of birth/death

        place_of_birth_raw = date_of_birth_raw.find("tei:placeName", NS) if date_of_birth_raw else None
        place_of_birth = place_of_birth_raw.text if place_of_birth_raw is not None else None

        place_of_death_raw = date_of_death_raw.find("tei:placeName", NS) if date_of_death_raw else None
        place_of_death = place_of_death_raw.text if place_of_death_raw is not None else None

        # Description
        description_raw = person_xml.find(".//tei:note/[@type='notes']", NS)
        description = description_raw.text if description_raw is not None else None

        # Faith
        faith_raw = person_xml.find("tei:faith", NS)
        faith = faith_raw.text if faith_raw is not None else None

        # Adding it
        person = Person(uid=xml_id, label=label, description=description, roles=roles,
                        alternative_labels=alternative_labels, date_of_birth=date_of_birth, date_of_death=date_of_death,
                        place_of_birth=place_of_birth, place_of_death=place_of_death, faith=faith)
        persons.append(person)

    return persons


def personography_to_rdf():
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
    wwo = Namespace("http://www.wwp.brown.edu/ns/1.0#")

    g.bind('skos', skos)
    g.bind('wwo', wwo)

    wwo_person = wwo.Person
    wwo_hasRoleName = wwo.hasRoleName
    wwo_date_of_birth = wwo.dateOfBirth
    wwo_date_of_death = wwo.dateOfDeath
    wwo_place_of_birth = wwo.placeOfBirth
    wwo_place_of_death = wwo.placeOfDeath
    wwo_faith = wwo.faith

    properties = [
        wwo_hasRoleName,
        wwo_date_of_birth,
        wwo_date_of_death,
        wwo_place_of_birth,
        wwo_place_of_death,
        wwo_faith,
    ]

    for prop in chain(properties):
        g.add((prop, RDF.type, RDF.Property))

    g.add((wwo_person, skos.broader, skos.Concept))

    for person in load_personography():
        s = URIRef(f"http://www.wwp.brown.edu/ns/1.0#{person.uid}")

        g.add((s, RDF.type, wwo_person))

        if person.label:
            g.add((s, skos.prefLabel, Literal(person.label)))

        if person.description:
            g.add((s, RDFS.comment, Literal(person.description)))

        for role in person.roles:
            g.add((s, wwo_hasRoleName, Literal(role)))

        for alt_label in person.alternative_labels:
            g.add((s, skos.altLabel, Literal(alt_label)))

        if person.date_of_birth:
            g.add((s, wwo_date_of_birth, Literal(person.date_of_birth)))

        if person.date_of_death:
            g.add((s, wwo_date_of_death, Literal(person.date_of_death)))

        if person.place_of_birth:
            g.add((s, wwo_place_of_birth, Literal(person.place_of_birth)))

        if person.place_of_death:
            g.add((s, wwo_place_of_death, Literal(person.place_of_death)))

        if person.faith:
            g.add((s, wwo_faith, Literal(person.faith)))

    with open(PATH_WWO_PERSONOGRAPHY_RDF, "wb") as f:
        f.write(g.serialize(format='turtle'))

if __name__ == '__main__':
    personography_to_rdf()
