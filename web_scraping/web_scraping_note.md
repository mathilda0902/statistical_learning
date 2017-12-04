# Web Scraping
- Realistically, data that you want to study won't always be available to you in the form of a curated data set.
- Need to go to the internets to find interesting data:
  - From an existing company
  - Text for NLP
  - Images

- SQL vs. Mongo
  - SQL - want to prevent redundancy in data by having tables with unique information and relations between them (normalized data).
    - Creates a framework for querying with joins.
    - Makes it easier to update database. Only ever have to change information in a single place.
    - This can result in "simple" queries being slower, but more complex queries are often faster.
  - Mongo - document based storage system. Does not enforce normalized data. Can have data redundancies in documents (denormalized data).
    - No joins.
    - A change to database generally results in needing to change many documents.
    - Since there is redundancy in the documents, simple queries are generally faster. But complex queries are often slower.
