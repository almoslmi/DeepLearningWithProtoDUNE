import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def plot_weights_median(weights, ranges, class_names, plot_name):
    """
    Plot weights for each class based on the frequencies of the samples and display median.
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(18,5), facecolor='w')
    for index in range(len(class_names)):
        ax = axes[index]
        ax.hist(weights[index], 100, range=ranges[index], color='green', alpha=0.75)
        ax.set_title(class_names[index], fontsize=20, fontname='Georgia',fontweight='bold')
        ax.set_xlabel("Weight", fontsize=15, fontname='Georgia',fontweight='bold')
        ax.set_ylabel("Count", fontsize=15, fontname='Georgia',fontweight='bold')

        _, max_ = ax.get_ylim()
        median = np.median(weights[index])
        ax.axvline(median, color='k', linestyle='dashed', linewidth=2)
        ax.text(median + median/10, max_ - max_/10, 'Median: {:.2f}'.format(median),
                fontsize=12, fontweight=1000, color='k')

    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')

def get_feature_image(feature_image, fig, ax, title):
    c = ax.imshow(feature_image, cmap='winter_r',interpolation='none', origin='lower',
                  norm=LogNorm(vmin=1.0, vmax=abs(feature_image).max()))
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Global wire no.", fontsize=15, fontname='Georgia',fontweight='bold')
    ax.set_ylabel("TDC", fontsize=1, fontname='Georgia',fontweight='bold')
    ax.set_title(title, fontsize=20,fontname='Georgia',fontweight='bold')

def get_label_image(label_image, fig, ax, title):
    minimum = np.min(label_image)
    maximum = np.max(label_image)
    cmap = plt.get_cmap('gist_heat_r', maximum-minimum+1)
    c= ax.imshow(label_image,cmap=cmap,interpolation='none', origin='lower',
                   vmin=minimum, vmax=maximum)
    cb = fig.colorbar(c, ax=ax)
    cb.set_ticks(categories_value)
    cb.set_ticklabels(categories_name)

    ax.set_xlabel("Global wire no.", fontsize=15, fontname='Georgia',fontweight='bold')
    ax.set_ylabel("TDC", fontsize=15, fontname='Georgia',fontweight='bold')
    ax.set_title(title, fontsize=20,fontname='Georgia',fontweight='bold')

def plot_feature_label(feature_image, label_image,
                       feature_title, label_tile,
                       plot_name):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,7), facecolor='w')

    get_feature_image(feature_image, fig, ax0, feature_title)
    get_label_image(label_image, fig, ax1, label_tile)

    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')

def plot_categories(feature_image, label_image, plot_name):
    fig, axes = plt.subplots(1, len(categories_value), figsize=(18,12), facecolor='w')
    for index, value in enumerate(categories_value):
        ax = axes[index]
        mask = (label_image == value)
        c = ax.imshow(feature_image*mask, cmap='winter_r',interpolation='none', origin='lower',
                  norm=LogNorm(vmin=1.0, vmax=np.max(feature_image)))
        ax.set_xlabel("Global wire no.", fontsize=15, fontname='Georgia',fontweight='bold')
        ax.set_ylabel("TDC", fontsize=1, fontname='Georgia',fontweight='bold')
        ax.set_title(categories_name[index], fontsize=20,fontname='Georgia',fontweight='bold')

    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')

def plot_feature_label_prediction(feature_image, label_image, prediction_image,
                                  feature_title, label_title, prediction_title,
                                  plot_name):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20,5), facecolor='w')

    get_feature_image(feature_image, fig, ax0, feature_title)
    get_label_image(label_image, fig, ax1, label_title)
    get_label_image(prediction_image, fig, ax2, prediction_title)

    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')
