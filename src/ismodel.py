import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np
import bz2
from sklearn.neighbors import NearestNeighbors

# load images and resize them to 100x100 to create a smaller model based on distance between image vector values
# model: kNN, 11 neighbors, euclidean distance
ds = keras.preprocessing.image_dataset_from_directory('static/',
                                                       label_mode=None,
                                                       class_names=None,
                                                       color_mode='rgb',
                                                       batch_size=1,
                                                       image_size=(100, 100),
                                                       shuffle=False,
                                                       seed=None,
                                                       validation_split=None,
                                                       subset=None,
                                                       interpolation='bilinear',
                                                       follow_links=False
                                                       )
images = np.array([img for img in ds]).reshape(4657, 100, 100, 3)
print(images[0])
features = []
for i in images:
    i = i/255
    features.append(i.flatten())
np_ds = np.array(features)
print(np_ds[0])
knn = NearestNeighbors(n_neighbors=11, metric='euclidean')
knn.fit(np_ds)

# testing on image
#image = images[10] # take an existing image or create a numpy array from PIL image
#feature = image/255
#feature = feature.flatten()
#feature = feature.reshape(1, -1)

#distances, nbors = knn.kneighbors(feature)
# output is a list of distances and neighbor index of each image in the model
# so we take the first entry from the output since we are evaluating only one image
#distances, nbors = distances[0], nbors[0]
#nbor_images = [images[i]/255 for i in nbors]
#fig, axes = plt.subplots(1, len(nbors)+1, figsize=(10, 10))
#
#for i in range(len(nbor_images)):
#    ax = axes[i]
#    ax.set_axis_off()
#    if i == 0:
#        ax.imshow(image/255)
#        ax.set_title("Input")
#    else:
#        ax.imshow(nbor_images[i])
#        # we get euclidean distance, to convert to similarity we do 1 - l1 norm of the distance
#        ax.set_title(f"Sim: {100*(1-(distances[i]/sum(distances))):.0f}%")
#plt.show()

import pickle
with bz2.BZ2File('isknn.pkl', 'wb') as f:
    pickle.dump(knn, f)
