import sys
import glob
import os
import argparse
import configparser
from tools.data_tools import get_data_generator
from tools.plotting_tools import plot_feature_label, plot_categories

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--events", required=True,
	   help="Choose number of events to run over.")
    return vars(ap.parse_args())

def main():
    args = argument_parser()
    try:
        NUM_EVENTS = int(args["events"])
        print("Running over {} events.".format(NUM_EVENTS))
    except ValueError:
        print("\nError: Events should be an integer.")
        print("Exiting!\n")
        sys.exit(1)

    config = configparser.ConfigParser()
    config_path = path.join("configurations", "master_configuration.ini")
    config.read(config_path)
    print("\nReading info from configuration:")

    IMAGE_WIDTH = int(config["DEFAULT"]["IMAGE_WIDTH"])
    IMAGE_HEIGHT = int(config["DEFAULT"]["IMAGE_HEIGHT"])
    IMAGE_DEPTH = int(config["DEFAULT"]["IMAGE_DEPTH"])
    CLASS_NAMES = config["DEFAULT"]["CLASS_NAMES"].split()

    FEATURE_FILE_TRAINING = config["DEFAULT"]["FEATURE_FILE_TRAINING"]
    LABEL_FILE_TRAINING = config["DEFAULT"]["LABEL_FILE_TRAINING"]
    FEATURE_FILE_VALIDATION = config["DEFAULT"]["FEATURE_FILE_VALIDATION"]
    LABEL_FILE_VALIDATION = config["DEFAULT"]["LABEL_FILE_VALIDATION"]
    FEATURE_FILE_TESTING = config["DEFAULT"]["FEATURE_FILE_TESTING"]
    LABEL_FILE_TESTING = config["DEFAULT"]["LABEL_FILE_TESTING"]

    print("IMAGE_WIDTH: {}".format(IMAGE_WIDTH))
    print("IMAGE_HEIGHT: {}".format(IMAGE_HEIGHT))
    print("IMAGE_DEPTH: {}".format(IMAGE_DEPTH))
    print("CLASS_NAMES: {}".format(CLASS_NAMES))

    print("FEATURE_FILE_TRAINING: {}".format(FEATURE_FILE_TRAINING))
    print("LABEL_FILE_TRAINING: {}".format(LABEL_FILE_TRAINING))
    print("FEATURE_FILE_VALIDATION: {}".format(FEATURE_FILE_VALIDATION))
    print("LABEL_FILE_VALIDATION: {}".format(LABEL_FILE_VALIDATION))
    print("FEATURE_FILE_TESTING: {}".format(FEATURE_FILE_TESTING))
    print("LABEL_FILE_TESTING: {}\n".format(LABEL_FILE_TESTING))
    print()

    generator_training = get_data_generator(FEATURE_FILE_TRAINING, LABEL_FILE_TRAINING)
    generator_validation = get_data_generator(FEATURE_FILE_VALIDATION, LABEL_FILE_VALIDATION)
    generator_testing = get_data_generator(FEATURE_FILE_TESTING, LABEL_FILE_TESTING)

    plot_path = os.path.join("plots",  "events", "*.pdf")
    files = glob.glob(plot_path)
    for f in files:
        os.remove(f)

    count = 0
    for X, y in generator_training:
        if count >= NUM_EVENTS:
            break
        count += 1

        feature_image = X.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        label_image = y.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        plot_feature_label_path = os.path.join("plots",  "events", "feature_label_training_event_{}.pdf".format(count))
        plot_feature_label(feature_image, label_image,  'Feature', 'Label', CLASS_NAMES, plot_feature_label_path)

        plot_categories_path = os.path.join("plots", "events", "categories_training_event_{}.pdf".format(count))
        plot_categories(feature_image, label_image, CLASS_NAMES, plot_categories_path)

    count = 0
    for X, y in generator_validation:
        if count >= NUM_EVENTS:
            break
        count += 1

        feature_image = X.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        label_image = y.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        plot_feature_label_path = path.join("plots",  "events", "feature_label_validaiton_event_{}.pdf".format(count))
        plot_feature_label(feature_image, label_image,  'Feature', 'Label', CLASS_NAMES, plot_feature_label_path)

        plot_categories_path = path.join("plots", "events", "categories_validation_event_{}.pdf".format(count))
        plot_categories(feature_image, label_image, CLASS_NAMES, plot_categories_path)

    count = 0
    for X, y in generator_testing:
        if count >= NUM_EVENTS:
            break
        count += 1

        feature_image = X.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        label_image = y.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        plot_feature_label_path = path.join("plots",  "events", "feature_label_testing_event_{}.pdf".format(count))
        plot_feature_label(feature_image, label_image,  'Feature', 'Label', CLASS_NAMES, plot_feature_label_path)

        plot_categories_path = path.join("plots", "events", "categories_testing_event_{}.pdf".format(count))
        plot_categories(feature_image, label_image, CLASS_NAMES, plot_categories_path)

    print("\nDone! Plots are saved in \plots!\n")

if __name__ == "__main__":
    main()
