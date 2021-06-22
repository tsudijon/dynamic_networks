import deepdish as dd
import glob

results = {}
for filename in glob.glob("*.h5"):
    res = dd.io.load(filename)
    for key in res:
        results[key] = res[key]
dd.io.save("results.h5", results)
