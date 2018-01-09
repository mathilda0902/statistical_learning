import boto

# Connect to S3 - DO NOT JUST DIRECTLY REPLACE THE BELOW WITH YOUR ACCESS
# AND SECRET ACCESS KEY!! This runs the risk of you pushing them
# up to github. Instead, read them into variables from a .json file or store
# them as environment variables in your .bashrc / .bash_profile and read
# them in from there. See your lecture notes or read_aws_credentials.md for how to do this.

conn = boto.connect_s3()

### If the above fails to authenticate, either create a file in ~/.aws with your credentials as
### described in read_aws_credentials.md, or use on of the other methods to load them from a file
### and use the code below
#access_key, access_secret_key = 'YOUR ACCESS KEY', 'YOUR SECRET ACCESS KEY'
#conn = boto.connect_s3(access_key, access_secret_key)



# List all the buckets
all_buckets = [b.name for b in conn.get_all_buckets()]
print all_buckets

# Check if bucket exist. If exist get bucket, else create one
bucket_name = 'multifacetedhotelrecommender'

if conn.lookup(bucket_name) is None:
    b = conn.create_bucket(bucket_name)
else:
    b = conn.get_bucket(bucket_name)

# Write new file
file_object = b.new_key('sample.txt')
file_object.set_contents_from_string('HAHAHAH')

# Read from file
print file_object.get_contents_as_string()

# Print all the files in the bucket
filenames = [f.name for f in b.list()]
print filenames

# Delete a file
a = b.new_key('somefilename')
a.delete()

# Delete Bucket
# Must delete all files in bucket
# Before deleting bucket
for key in b.get_all_keys():
    key.delete()
conn.delete_bucket(bucket_name)
