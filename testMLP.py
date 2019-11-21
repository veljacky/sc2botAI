from glob import glob
import numpy as np
import pickle
import datetime
import neuralnetworkv2
from keras.callbacks import TensorBoard, ModelCheckpoint


def data_to_train(data):
    structured, visual,  label = [], [], []
    for elem in data:
        struct = np.array(elem[0], dtype="float64")
        structured.append(struct)
        visual.append(elem[1])
        label.append(int(elem[2]))

    return np.array(structured), np.array(visual), np.array(label)


DATA_PATH = "balancedData\\"

replays = glob(DATA_PATH + "*.pickle")

dataset = []
for i, path in enumerate(replays):
    print("Currenty working on {} file!".format(i))
    with open(path, "rb") as f:
       ds = pickle.load(f)
    dataset.extend(ds)

ds, visual, labels = data_to_train(dataset)
print(ds.shape)
train_ds, valid_ds = ds[:int(0.8*ds.shape[0]), :], ds[int(0.8*ds.shape[0]):, :]
train_labels, valid_labels = labels[:int(0.8*ds.shape[0])], labels[int(0.8*ds.shape[0]):]
visual_ds, visual_valid_ds = visual[:int(0.8*ds.shape[0])], visual[int(0.8*ds.shape[0]):]


log_dir=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

filename = "neuralnetwork\\\models\\best_weights.hdf5"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = ModelCheckpoint(filename, monitor="val_accuracy", mode="max", save_best_only=True, verbose=True)

#model = neuralnetworkv2.create_mlp(test=True)
model = neuralnetworkv2.create_final_model()
history = model.fit([train_ds, visual_ds], train_labels, validation_data=[[valid_ds, visual_valid_ds], valid_labels], epochs=60, callbacks=[tensorboard_callback, checkpoint])
#history = model.fit([train_ds], train_labels, validation_data=[[valid_ds], valid_labels], epochs=200, callbacks=[tensorboard_callback, checkpoint])

# acc = model.evaluate(*test_data, test_labels)
model.save("trained2.h5")

