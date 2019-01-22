import zipfile

with zipfile.ZipFile("test.zip", "w") as myzipfile:
    myzipfile.write("test.txt")

with zipfile.ZipFile("test.zip", "r") as myzipfile:
    myzipfile.extractall()
