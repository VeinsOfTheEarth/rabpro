from email import header
import pandas as pd

dt = pd.read_json("Data/user_gee_datasets.json")

dt = dt[["name", "id"]]

res = "`" + dt.name + " <https://code.earthengine.google.com/?asset=" + dt.id + ">`_"

res.to_csv("docs/source/user_gee_datasets.csv", index=False, header=False)

