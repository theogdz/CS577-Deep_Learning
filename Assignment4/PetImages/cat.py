from os import listdir, rename
from os.path import isfile, join
mypath = 'cat'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
arr = []
for a in onlyfiles:
    arr.append(a[:-4])   
arr.sort(key=int)
print(arr[:10])
for i,x in enumerate(arr):
    rename("cat\\"+x+".jpg","cat\\"+str(i)+".jpg")