# acl2020-interactive-entity-linking

## Data preparation

### 1641

The following steps describe how to convert the depositions NIF file to documents and how to generate the knowledge base. The data comes from https://github.com/munnellg/1641DepositionsCorpus

1. Import the `pom.xml` in `data-converter` in your favorite Java IDE, I like IntelliJ IDEA. This should automatically download all dependencies.
2. Run `de.tudarmstadt.ukp.gleipnir.depositions.App1641`
3. The corpus should be written to `linker/generated/depositions` and be already be split

### WWO