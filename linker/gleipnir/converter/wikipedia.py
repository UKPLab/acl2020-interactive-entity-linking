import csv
import gzip
import sqlite3
from string import Template
import sys
from typing import Optional

from urllib.parse import quote

from gleipnir.config import *


def _is_insert(line):
    """
    Returns true if the line begins a SQL insert statement.
    """
    return line.startswith('INSERT INTO')


def _get_values(line):
    """
    Returns the portion of an INSERT statement containing values
    """
    return line[line.find("` VALUES ") + 9:]


def _values_sanity_check(values):
    """
    Ensures that values from the INSERT statement meet basic checks.
    """
    assert values
    assert values[0] == '('
    # Assertions have not been raised
    return True


def _parse_values(values):
    """
    Given a file handle and the raw values from a MySQL INSERT
    statement, write the equivalent CSV to the file
    """
    latest_row = []

    reader = csv.reader([values], delimiter=',',
                        doublequote=False,
                        escapechar='\\',
                        quotechar="'",
                        strict=True
    )

    for reader_row in reader:
        for column in reader_row:
            # If our current string is empty...
            if len(column) == 0 or column == 'NULL':
                latest_row.append(chr(0))
                continue
            # If our string starts with an open paren
            if column[0] == "(":
                # Assume that this column does not begin
                # a new row.
                new_row = False
                # If we've been filling out a row
                if len(latest_row) > 0:
                    # Check if the previous entry ended in
                    # a close paren. If so, the row we've
                    # been filling out has been COMPLETED
                    # as:
                    #    1) the previous entry ended in a )
                    #    2) the current entry starts with a (
                    if latest_row[-1][-1] == ")":
                        # Remove the close paren.
                        latest_row[-1] = latest_row[-1][:-1]
                        new_row = True
                # If we've found a new row, write it out
                # and begin our new one
                if new_row:
                    yield latest_row
                    latest_row = []
                # If we're beginning a new row, eliminate the
                # opening parentheses.
                if len(latest_row) == 0:
                    column = column[1:]
            # Add our column to the row we're working on.
            latest_row.append(column)
        # At the end of an INSERT statement, we'll
        # have the semicolon.
        # Make sure to remove the semicolon and
        # the close paren.
        if latest_row[-1][-2:] == ");":
            latest_row[-1] = latest_row[-1][:-2]
            yield latest_row


def create_wikipedia_wikidata_mapping_db():
    """ Uses a Wikipedia dump to construct a mapping between Wikidata and Wikipedia.
        Credit: Uses parts of https://github.com/jamesmishra/mysqldump-to-csv
    """

    csv.field_size_limit(sys.maxsize)

    # Create the database file
    os.makedirs(os.path.dirname(PATH_WIKIPEDIA_WIKIDATA_INDEX), exist_ok=True)

    try:
        os.remove(PATH_WIKIPEDIA_WIKIDATA_INDEX)
    except FileNotFoundError:
        pass

    conn = sqlite3.connect(PATH_WIKIPEDIA_WIKIDATA_INDEX, isolation_level="EXCLUSIVE")

    with conn:
        conn.execute('''CREATE TABLE mapping (
            wikipedia_id int PRIMARY KEY ,
            wikipedia_name text,
            wikidata_id text)''')
        conn.execute('''CREATE UNIQUE INDEX idx_wikipedia_name ON mapping(wikipedia_name);''')

    c = conn.cursor()

    print("Wikipedia page")

    # Parse the Wikipedia page dump; extract page id and page title from the sql
    # https://www.mediawiki.org/wiki/Manual:Page_table
    with gzip.open(PATH_WIKIPEDIA_PAGES_RAW, "rt", encoding="utf-8", newline="\n") as f:
        for line in f:
            # Look for an INSERT statement and parse it.
            if not _is_insert(line):
                continue

            values = _get_values(line)

            for v in _parse_values(values):
                # Filter the namespace; only use real articles
                # https://www.mediawiki.org/wiki/Manual:Namespace
                if v[1] == "0":
                    c.execute("INSERT INTO mapping (wikipedia_id, wikipedia_name) VALUES (?, ?)", (v[0], v[2]))

    conn.commit()

    print("Wikipedia page property")

    # Parse the Wikipedia page property dump; extract page id and Wikidata id from the sql
    # https://www.mediawiki.org/wiki/Manual:Page_props_table/en
    # Parse the Wikipedia page property dump; extract page id and Wikidata id from the sql
    with gzip.open(PATH_WIKIPEDIA_PAGE_PROPS_RAW, "r") as f:
        for line in f:
            line = line.decode("utf-8", "ignore")

            # Look for an INSERT statement and parse it.
            if not _is_insert(line):
                continue

            values = _get_values(line)
            for v in _parse_values(values):
                # The page property table contains many properties, we only care about the Wikidata id
                if v[1] == "wikibase_item":
                    c.execute(f"UPDATE mapping SET wikidata_id = ? WHERE wikipedia_id = ?", (v[2], v[0]))
    conn.commit()

    # Parse the Wikipedia redirect dump; fill in missing Wikidata ids
    # https://www.mediawiki.org/wiki/Manual:Redirect_table
    with gzip.open(PATH_WIKIPEDIA_REDIRECTS_RAW, "rt", encoding="utf-8", newline="\n") as f:
        for line in f:
            # Look for an INSERT statement and parse it.
            if not _is_insert(line):
                continue

            values = _get_values(line)

            for v in _parse_values(values):
                source_wikipedia_id = v[0]
                target_title = v[2]
                namespace = v[1]

                # We only care about main articles
                if namespace != "0":
                    continue

                c.execute(f"SELECT wikidata_id FROM mapping WHERE wikipedia_name = ?", (target_title,))
                result = c.fetchone()

                if result is None or result[0] is None:
                    continue

                wikidata_id = result[0]
                c.execute(f"UPDATE mapping SET wikidata_id = ? WHERE wikipedia_id = ?",
                          (wikidata_id, source_wikipedia_id))

    conn.commit()

    c.close()
    conn.close()


def map_wikipedia_url_to_wikidata_url(wiki_url: str) -> str:
    """ Given a Wikipedia URL, returns the corresponding Wikidata IRI.
        Uses a precomputed database created by `create_wikipedia_wikidata_mapping_db`.
    """
    if wiki_url is "NIL":
        return "NIL"

    title = wiki_url.rsplit('/', 1)[-1]

    with sqlite3.connect(PATH_WIKIPEDIA_WIKIDATA_INDEX) as conn:
        c = conn.cursor()
        c.execute('SELECT wikidata_id FROM mapping WHERE wikipedia_name=?', (title, ))
        result = c.fetchone()

    if result is not None and result[0] is not None:
        return f"http://www.wikidata.org/entity/{result[0]}"
    else:
        print(wiki_url)
        return "NIL"

