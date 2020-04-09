# acl2020-interactive-entity-linking

## General setup

venv
requirements    

## Data preparation

### AIDA

### 1641

The following steps describe how to convert the depositions NIF file to documents and how to generate the knowledge base. The data comes from [here](https://github.com/munnellg/1641DepositionsCorpus) . We use DKPro for the NIF to text conversion. Please refer to the supplementary material for further explanations regarding preprocessing.

1. Import the `pom.xml` in `data-converter` in your favorite Java IDE, I like IntelliJ IDEA. This should automatically download all dependencies.
2. Run `de.tudarmstadt.ukp.gleipnir.depositions.App1641`
3. The corpus should be written to `linker/generated/depositions` and be already be split
4. Run `linker/gleipnir/converter/depositions.py` to generate the knowledge base. The file is written to `linker/generated/depositions/kb/depositions_kb.ttl`

### WWO

The following steps describe how to convert the WWO data to documents and how to generate the knowledge base. We use DKPro for the TEI to text conversion. Please refer to the supplementary material for further explanations regarding preprocessing.

1. Copy the WWO data you obtained from the [Womens Writer project](https://www.wwp.northeastern.edu/) to `data-converter/wwo`. It should look like the following:

    $ data-converter/wwo: ls -l
    README
    aggregate
    common-boilerplate.xml
    files
    personography.xml
    persons.diff
    schema
    words

2. Run `de.tudarmstadt.ukp.gleipnir.wwo.AppWwo`
3. The corpus should be written to `linker/generated/wwo` and already be split
4. Run `linker/gleipnir/converter/wwo.py` to generate the WWO knowledge base from the personography. It will be written to `linker/generated/wwo/personography.ttl`.