## MongoDB Cheat Sheet

#### Connect to a database after starting mongo client
```
$ mongo
use dbname
```
> Call this command before using a specific database.
> This command also creates the database, but the new database is only save when you insert the first document in a collection.

#### Connect to a particular database when starting mongo client
```sh
$ mongo dbname
```

#### Drop a particular database
```
use dbname
db.dropDatabase()
```

#### List all databases
```js
show databases
```

#### List all collections of a database
```js
use dbname
show collections
```

#### Get the status of a particular database
```
db.status()
```
> As of version 3.2.10, this commands lists an object like the following:

```
{
  "db" : "dbname",
  "collections" : 0,
  "objects" : 0,
  "avgObjSize" : 0,
  "dataSize" : 0,
  "storageSize" : 0,
  "numExtents" : 0,
  "indexes" : 0,
  "indexSize" : 0,
  "fileSize" : 0,
  "ok" : 1
}
```
> By default the data size will show up in bytes, but if you want it in KB, use this `db.stats(1024)`. For MB, use `db.stats(1024*1024)`

#### List the current connections to a mongodb server
```js
db.serverStatus().connections
```

-----------------------

## CRUD Operations
#### Insert a document
```js
db.movies.insertOne({"title":"New Movie", "year":2010, "imdb": "aa0099999"})
```
> This command inserts a document into a movies collection

#### Count the number of documents in a given collection
```js
db.movies.count()
```

#### List documents in a collection without specifying any constraints
```js
db.movies.find()
// or if you want to format the output
db.movies.find().pretty()
```

#### List documents in a collection using a cursor
```js
// The mongo shell talks JavaScript
var cs = db.movies.find()
cs.hasNext() // checks whether or not there is a document to show
cs.next() // outputs the next object
// You could keep on doing this until you exaust the cursor, but it is problably not a good idea if the queries returns thousands of documents. rsrsrsr
// This techinique is very useful when working with a driver such as the Java driver
```

#### Finding and Sorting
- `sort({field: 1|-1})`: 1 = ascending; -1 = descending

List subdocuments ordered by `subdoc.field`. It returns only a list with 10 docs, leaving out all other fields, even `_id`.
```js
db.collection.find({},{_id:null, "subdoc.field":1}).sort({"subdoc.field": -1}).limit(10).pretty()
```

---------------------------------
### Aggregation
```js
// things collection
{
  name: "John",
  country: "US",
  children: [{
    name: "Rachel",
    age: 10,
    watched_movies: [{
      title: "movie 1"
    },
    {
      title: "movie 2"
    }]
  }]
}
```
Given the above document, how would you get the number of watched movies by John's children?
```
db.things.aggregate([
	{ $match: { name: "John" } },
	{ $unwind: "$children" },
	{ $unwind: "$children.watched_movies" },
	{ $group: { _id: null, count: { $sum: 1 } } }
])
```
Get a cursor, add all documents in an array and call `tojson()` to print them
```js
records = [];
var cursor = db.someCollection.find({}, {}).limit(100);
while(cursor.hasNext()) {
    records.push(cursor.next())
    //var c = cursor.next();
    //print(c.field1 + ',' + c.field2 + ',' + c.field3)
}
print(tojson(records))
```


-----------------------------------

## Dump, Import, Export
#### Import a JSON file containing a collection of companies into a database called mclass
```sh
sudo mongoimport --db mclass --collection companies --file companies.json
```

----------------------

### Useful links
- [How To Import and Export a MongoDB Database on Ubuntu 14.04](https://www.digitalocean.com/community/tutorials/how-to-import-and-export-a-mongodb-database-on-ubuntu-14-04) 
