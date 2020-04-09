# acl2020-interactive-entity-linking

## Data preparation

### 1641

The following steps describe how to convert the depositions NIF file to documents and how to generate the knowledge base. The data comes from https://github.com/munnellg/1641DepositionsCorpus . We use DKPro for the NIF to text conversion. Please refer to the supplementary material for further explanations regarding preprocessing.

1. Import the `pom.xml` in `data-converter` in your favorite Java IDE, I like IntelliJ IDEA. This should automatically download all dependencies.
2. Run `de.tudarmstadt.ukp.gleipnir.depositions.App1641`
3. The corpus should be written to `linker/generated/depositions` and be already be split
4. Knowledge base

### WWO

The following steps describe how to convert the WWO data to documents and how to generate the knowledge base. We use DKPro for the TEI to text conversion. Please refer to the supplementary material for further explanations regarding preprocessing.

1. Copy the WWO data you obtained from the Womens Writer project to `data-converter/wwo`. It should look like the following:

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
3. The corpus should be written to `linker/generated/wwo` and be already be split
4. Knowledge base