import argparse
import configparser
import sys
#import get_data

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--operation", required=True,
	   help="Choose operation between 'Training' or 'Development'")
    ap.add_argument("-e", "--epoch", required=True,
	   help="Options are 'Default' or a number")
    return vars(ap.parse_args())

def main():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    print("\nReading info from configuration:")

    args = argument_parser()
    if args["operation"] == "Training":
        print("\nRunnning in training setting!\n")

        NUM_TRAINING = int(config["TRAINING"]["NUM_TRAINING"])
        NUM_VALIDATION = int(config["TRAINING"]["NUM_VALIDATION"])
        NUM_TESTING = int(config["TRAINING"]["NUM_TESTING"])
        NUM_EPOCHS = int(config["TRAINING"]["NUM_EPOCHS"])

    elif args["operation"] == "Development":
        print("\nRunnning in development setting!\n")

        NUM_TRAINING = int(config["DEVELOPMENT"]["NUM_TRAINING"])
        NUM_VALIDATION = int(config["DEVELOPMENT"]["NUM_VALIDATION"])
        NUM_TESTING = int(config["DEVELOPMENT"]["NUM_TESTING"])
        NUM_EPOCHS = int(config["DEVELOPMENT"]["NUM_EPOCHS"])

    else:
        print("\nError: Operation should be either 'Training' or 'Development'")
        print("Exiting!\n")
        sys.exit(1)

    if args["epoch"] != "Default":
        try:
            NUM_EPOCHS = int(args["epoch"])
        except ValueError:
            print("\nError: Epoch should be an integer.")
            print("Exiting!\n")
            sys.exit(1)

    BATCH_SIZE = config["DEFAULT"]["BATCH_SIZE"]
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
    WEIGHTS = list(map(float, config["DEFAULT"]["WEIGHTS"].split()))

    print("NUM_TRAINING: {}".format(NUM_TRAINING))
    print("NUM_VALIDATION: {}".format(NUM_VALIDATION))
    print("NUM_TESTING: {}".format(NUM_TESTING))
    print("NUM_EPOCHS: {}".format(NUM_EPOCHS))
    print("BATCH_SIZE: {}".format(BATCH_SIZE))
    print("IMAGE_WIDTH: {}".format(IMAGE_WIDTH))
    print("IMAGE_HEIGHT: {}".format(IMAGE_HEIGHT))
    print("IMAGE_DEPTH: {}".format(IMAGE_DEPTH))
    print("CLASS_NAMES: {}".format(CLASS_NAMES))
    print("FEATURE_FILE_TRAINING: {}".format(FEATURE_FILE_TRAINING))
    print("LABEL_FILE_TRAINING: {}".format(LABEL_FILE_TRAINING))
    print("FEATURE_FILE_VALIDATION: {}".format(FEATURE_FILE_VALIDATION))
    print("LABEL_FILE_VALIDATION: {}".format(LABEL_FILE_VALIDATION))
    print("FEATURE_FILE_TESTING: {}".format(FEATURE_FILE_TESTING))
    print("LABEL_FILE_TESTING: {}".format(LABEL_FILE_TESTING))
    print("WEIGHTS: {}".format(WEIGHTS))



if __name__ == "__main__":
    main()
