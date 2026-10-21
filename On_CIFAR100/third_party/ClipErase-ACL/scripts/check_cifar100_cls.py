import pickle


with open("./cifar-100-python/meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")

label_names: dict = {}
for i, name in enumerate(meta[b"fine_label_names"]):
    label_names[i] = name.decode("utf-8")

print(label_names)
