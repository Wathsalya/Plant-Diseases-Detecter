_URL = "https://github.com/AveyBD/rice-leaf-diseases-detection/raw/master/rice-leaf.zip"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="rice-leaf.zip",
                                   extract=True)
print(zip_file)
# C:\Users\user1\.keras\datasets\flower_photos.tgz

print(os.path.dirname(zip_file))

# C:\Users\user1\.keras\datasets

base_dir = os.path.join(os.path.dirname(zip_file), 'rice')