# imageSimilarity-TimMcAnany
Assessment

docker image: https://hub.docker.com/repository/docker/mcantimmy1/imagesimilarityxgen

model used: ismodel.py -> isknn.pkl

flask api location: localhost:5000

image location: static/uploads

app: main.py - asks for an image to be selected, runs the kNN model on the given image, and renders list of 10 most similar images in the given set of product images
