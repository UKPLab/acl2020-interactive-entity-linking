# From Zero to Hero: Human-In-The-Loop Entity Linking in Low Resource Domains

### Jan-Christoph Klie, Richard Eckart de Castilho and Iryna Gurevych
#### [UKP Lab, TU Darmstadt](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp)

Source code for our experiments of our [ACL 2020 article](to appear).

> **Abstract:** Entity linking (EL) is concerned with disambiguating entity mentions in a text against knowledge bases (KB). It is crucial in a considerable number of fields like humanities, technical writing and biomedical sciences to enrich texts with semantics and discover more knowledge. The use of EL in such domains requires handling noisy texts, low resource settings and domain-specific KBs. Existing approaches are mostly inappropriate for this, as they depend on training data. However, in the above scenario, there exists hardly annotated data, and it needs to be created from scratch. We therefore present a novel domain-agnostic Human-In-The-Loop annotation approach: we use recommenders that suggest potential concepts and adaptive candidate ranking, thereby speeding up the overall annotation process and making it less tedious for users. We evaluate our ranking approach in a simulation on difficult texts and show that it greatly outperforms a strong baseline in ranking accuracy. In a user study, the annotation speed improves by 35 % compared to annotating without interactive support; users report that they strongly prefer our system.

* **Contact person:** Jan-Christoph Klie, klie@ukp.informatik.tu-darmstadt.de
    * UKP Lab: http://www.ukp.tu-darmstadt.de/
    * TU Darmstadt: http://www.tu-darmstadt.de/

Drop me a line or report an issue if something is broken (and shouldn't be) or if you have any questions.

For license information, please see the `LICENSE` and `README` files.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure

This repository contains to projects, `data-converter` is a Java application for converting the data and `linker`, a Python project which contains all relevant experiments.

## Setting up the experiments

In the `linker` folder, run

    pip install -r requirements.txt

Extract the `zero_to_hero.zip` from [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2316) into `linker/generated` .

## Running experiments

### Evaluate Ranker

In order to evaluate the different ranker on the full train/dev/test split, change the models and datasets you want to evaluate and run `linker/scripts/evaluate_ranking.py`.

### Simulation

Change the models and datasets you want to evaluate, then run `linker/scripts/simulation.py`. The results can be found under `linker/results/${TIMESTAMP}`.

## Data preparation

This is only needed if you want to recreate the `1641` and `WWO` dataset. For most cases, use the dataset provided.

### Dataset preprocessing

#### AIDA

TBD due to licensing reasons

#### 1641

The following steps describe how to convert the depositions NIF file to documents and how to generate the knowledge base. The data comes from [here](https://github.com/munnellg/1641DepositionsCorpus) . We use DKPro for the NIF to text conversion. Please refer to the supplementary material for further explanations regarding preprocessing.

1. Import the `pom.xml` in `data-converter` in your favorite Java IDE, I like IntelliJ IDEA. This should automatically download all dependencies.
2. Run `de.tudarmstadt.ukp.gleipnir.depositions.App1641`
3. The corpus should be written to `linker/generated/depositions` and be already be split
4. Run `linker/gleipnir/converter/depositions.py` to generate the knowledge base. The file is written to `linker/generated/depositions/kb/depositions_kb.ttl`

#### WWO

The following steps describe how to convert the WWO data to documents and how to generate the knowledge base. We use DKPro for the TEI to text conversion. Please refer to the supplementary material for further explanations regarding preprocessing.

1. Copy the WWO data you obtained from the [Women Writers project](https://www.wwp.northeastern.edu/) to `data-converter/wwo`. It should look like the following:

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

### Knowledge base setup

#### Fuseki

For WWO and 1641, we use [Fuseki](https://jena.apache.org/documentation/fuseki2/) as the knowledge base server. To setup, follow the following steps:

1. Download Apache Jena and Fuseki 3.12.0 from the [project page](http://archive.apache.org/dist/jena/binaries/). The version is important.
2. Build the search index by running `build_index.sh` in the `fuseki` folder.
3. The knowledge base then can be started by running `run_fuseki.sh 1641` or `run_fuseki.sh wwo`

### Generate datasets

We precompute datasets and their features before experiments. Adjust which datasets you want to run. We cache requests to knowledge bases in order to make it feasible and not stress the endpoint too much. If you run the wrong KB with a dataset, remove the cache folder. Start the knowledge base as described in `Knowledge base setup`. Create datasets by running`linker/scripts/create_dataset.py`.

## Acknowledgements

This project uses data from [DBPedia](https://wiki.dbpedia.org/) for `1641`. Please refer to their license when using the generated data in this project. All rights for the `WWO` data are by the [Women Writers project](https://www.wwp.northeastern.edu/).



