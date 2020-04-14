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

## Knowledge base setup

### Fuseki

For WWO and 1641, we use fuseki as the knowledge base server. To setup, follow the following steps:

1. Download Apache Jena and Fuseki 3.12.0 from the [project page](http://archive.apache.org/dist/jena/binaries/). The version is important.
2. Build the search index by running `build_index.sh` in the `fuseki` folder.
3. The knowledge base then can be started by running `run_fuseki.sh 1641` or `run_fuseki.sh wwo`

## Generate datasets

We precompute datasets and their features before experiments. Adjust which datasets you want to run. We cache requests to knowledge bases in order to make it feasible and not stress the endpoint too much. If you run the wrong KB with a dataset, remove the cache folder. Start the knowledge base as described in `Knowledge base setup`. Create datasets by running`linker/gleipnir/handcrafted.py`.

## Running experiments

### Evaluate Ranker

In order to evaluate the different ranker on the full train/dev/test split, change the models and datasets you want to evaluate and run `linker/gleipnir/experiments/evaluate_ranking.py`.

### Simulation

Change the models and datasets you want to evaluate, then run `linker/gleipnir/experiments/simulation.py`. The results can be found under `linker/results/${TIMESTAMP}`.

