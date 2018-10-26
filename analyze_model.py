import sys
import argparse
import configparser
from os import path
from pickle import load
from tools.data_tools import get_data_generator
from tools.plotting_tools import plot_history

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--events", required=True,
	   help="Choose number of events to run over.")
    return vars(ap.parse_args())

def main():
    args = argument_parser()
    try:
        NUM_EVENTS = int(args["events"])
        print("Running over {} testing events.".format(NUM_EVENTS))
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

    FEATURE_FILE_TESTING = config["DEFAULT"]["FEATURE_FILE_TESTING"]
    LABEL_FILE_TESTING = config["DEFAULT"]["LABEL_FILE_TESTING"]

    print("IMAGE_WIDTH: {}".format(IMAGE_WIDTH))
    print("IMAGE_HEIGHT: {}".format(IMAGE_HEIGHT))
    print("IMAGE_DEPTH: {}".format(IMAGE_DEPTH))
    print("CLASS_NAMES: {}".format(CLASS_NAMES))
    print("FEATURE_FILE_TESTING: {}".format(FEATURE_FILE_TESTING))
    print("LABEL_FILE_TESTING: {}\n".format(LABEL_FILE_TESTING))
    print()

    # Get the history
    history_path = path.join("saved_models", "history.pkl")
    history = load(open(history_path, 'rb'))

    # Plot the history
    #hisotry_plots_path = path.join("plots", "events", "categories_validation_event_{}.pdf".format(count))
    #plot_history(history):

    print(history.keys())

if __name__ == "__main__":
    main()
