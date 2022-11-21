import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np
#import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

ds = keras.preprocessing.image_dataset_from_directory('image/',
                                                       label_mode=None,
                                                       class_names=None,
                                                       color_mode='rgb',
                                                       batch_size=1,
                                                       image_size=(100, 100),
                                                       shuffle=True,
                                                       seed=None,
                                                       validation_split=None,
                                                       subset=None,
                                                       interpolation='bilinear',
                                                       follow_links=False
                                                       )
images = np.array([img for img in ds]).reshape(6909, 100, 100, 3)
print(images[0])
features = []
for i in images:
    i = i/255
    #arrays = [j[:][:][0], j[:][:][1], j[:][:][2]]
    #j = np.prod(np.array(arrays), axis=0)
    features.append(i.flatten())
np_ds = np.array(features)
print(np_ds[0])
knn = NearestNeighbors(n_neighbors=11, metric = 'euclidean')
knn.fit(np_ds)

image = images[10] # take an existing image or create a numpy array from PIL image
#image = np.expand_dims(image, 0) # add a batch dimension
print(image)
feature = image/255
#feature = np.prod(np.array([feature[:][:][0], feature[:][:][1], feature[:][:][2]]), axis=0)
feature = feature.flatten()
print(feature)
feature = feature.reshape(1, -1)

distances, nbors = knn.kneighbors(feature)
# output is a tuple of list of distances and list nbors of each image
# so we take the first entry from those lists since we have only one image
distances, nbors = distances[0], nbors[0]
nbor_images = [images[i]/255 for i in nbors] # remove the first image since it is the same image
fig, axes = plt.subplots(1, len(nbors)+1, figsize=(10, 10))

for i in range(len(nbor_images)):
    ax = axes[i]
    ax.set_axis_off()
    if i == 0:
        ax.imshow(image/255)
        ax.set_title("Input")
    else:
        ax.imshow(nbor_images[i])
        # we get cosine distance, to convert to similarity we do 1 - cosine_distance
        ax.set_title(f"Sim: {100*(1-(distances[i]/sum(distances))):.0f}%")
plt.show()

import pickle
with open('isknn.pkl', 'wb') as f:
    pickle.dump(knn, f)
#vectorizer = keras.Sequential([
#    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5", trainable=False)
#])
#vectorizer.build([None, 256, 256, 3])
#
#features = vectorizer.predict(ds, batch_size=1)
#print(features.shape) # (640, 1536)