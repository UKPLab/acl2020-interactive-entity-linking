# https://medium.com/@rrichajalota234/how-to-apache-jena-fuseki-3-x-x-1304dd810f09

JENA_HOME=apache-jena-3.12.0
FUSEKI_HOME=apache-jena-fuseki-3.12.0

APACHE_JENA_BIN=./$JENA_HOME/bin
JENA_FUSEKI_JAR=./$FUSEKI_HOME/fuseki-server.jar

index_kb () {
    data_folder=$1
    path_to_ttl=$2
    path_to_config=$3
    rm -rf $data_folder
    mkdir -p $data_folder

    $APACHE_JENA_BIN/tdbloader2 --loc=$data_folder/tdb $path_to_ttl
    java -cp $JENA_FUSEKI_JAR jena.textindexer --desc $path_to_config

    rm -rf "$FUSEKI_HOME/$data_folder"
    cp -r $data_folder $FUSEKI_HOME/$data_folder
}

index_kb data_wwo ../generated/wwo/personography.ttl ./config_wwo.ttl
index_kb data_depositions ../generated/depositions/kb/depositions_kb.ttl ./config_1641.ttl
