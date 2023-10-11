import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

def pca_and_plot(code, degs, name, idx, path, forTitle, category):

    pca = PCA(n_components=2)
    red = pca.fit_transform(code)

    pca1, pca2 = zip(*red)

    plt.figure(idx)
    plt.scatter(pca1, pca2, s = 2,  c=degs)
    cbar = plt.colorbar()
    cbar.set_label("In-plane Rotations")

    plt.title("PCA {}\nCategory: {}".format(forTitle,category), weight = 'bold')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "{}.png".format(name)))

def tsne_and_plot(code, degs, name, idx, path,  forTitle, category):

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    red = tsne.fit_transform(code)

    tsne1, tsne2 = zip(*red)

    plt.figure(idx)
    plt.scatter(tsne1, tsne2, s = 2,  c=degs)
    cbar = plt.colorbar()
    cbar.set_label("In-plane Rotations")

    plt.title("t-SNE {}\nCategory: {}".format(forTitle,category), weight = 'bold')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "{}.png".format(name)))

categories = ["bowl", "camera", "mug", "laptop"]

checkpoint_dir='log'
logname='exp23-kaifAUG_bowl'
category = [cat for cat in categories if cat in logname]
if not category:
    category = 'bottle'
else:
    category = category[0]
print(category)
test = 'TestMixRot'
path="/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/{}/{}/{}/".format(checkpoint_dir, logname, test)

with open(os.path.join(path, "_pca_data.pkl"), "rb") as file:
            data = pickle.load(file)

img_code = np.array(data["img_code"])
shape_code = np.array(data["shape_code"])
degs = data["degs"]
img_paths = data["imgs_path"]

pca_and_plot(img_code, degs, "_pca_imgcode", 0, path, "512-D Feature Vector", category)
pca_and_plot(shape_code, degs, "_pca_shapecode", 1, path, "64-D Implicit Shape Code", category)

# tsne_and_plot(img_code, degs, "_tsne_imgcode", 0, path, "512-D Feature Vector", category)
# tsne_and_plot(shape_code, degs, "_tsne_shapecode", 1, path, "Ushape", category)
