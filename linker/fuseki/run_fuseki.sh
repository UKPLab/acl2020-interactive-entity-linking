name=$1
cd apache-jena-fuseki-3.12.0
java -Xmx32G -jar fuseki-server.jar --conf=../config_"${name}".ttl
