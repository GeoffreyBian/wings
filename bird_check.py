import os

train_classes = set(os.listdir("more_data/train"))
val_classes = set(os.listdir("more_data/valid"))

print("Only in train:", train_classes - val_classes)
print("Only in val:", val_classes - train_classes)
