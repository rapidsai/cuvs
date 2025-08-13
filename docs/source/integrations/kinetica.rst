Kinetica
--------

Starting with release 7.2, Kinetica supports the graph-based the CAGRA algorithm from RAFT. Kinetica will continue to improve its support over coming versions, while also migrating to cuVS as we work to move the vector search algorithms out of RAFT and into cuVS.

Kinetica currently offers the ability to create a CAGRA index in a SQL `CREATE_TABLE` statement, as outlined in their `vector search indexing docs <https://docs.kinetica.com/7.2/concepts/indexes/#cagra-index>`_. Kinetica is not open source, but the RAFT indexes can be enabled in the developer edition, which can be installed `here <https://www.kinetica.com/try/#download_instructions>`_.
