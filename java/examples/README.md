Building and Running
--------------------

Make sure to have JDK 22 and Maven 3.9.6+.

    mvn clean compile assembly:single

    java --enable-native-access=ALL-UNNAMED -jar ./target/cagra-sample-1.0-SNAPSHOT-jar-with-dependencies.jar
