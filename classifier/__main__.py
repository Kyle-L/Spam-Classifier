import argparse
from spam import *

class console_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Classify
    parser.add_argument('TrainingSetPath',
                        metavar='training-set-path',
                        type=str,
                        help='The path to the training data set.')
    parser.add_argument('TestingSetPath',
                        metavar='testing-set-path',
                        type=str,
                        help='The path to the test data set.')


    args = parser.parse_args()

    training_path = args.TrainingSetPath
    test_path = args.TestingSetPath
    
    try: 
        print(console_colors.HEADER + f'Training classifier with training set: {training_path}' + console_colors.ENDC)
        classifier = Classifier(training_path)
        print(console_colors.HEADER + f'Classify with test set: {test_path}' + console_colors.ENDC)
        accuracy = classifier.classify_test_set(test_path)
        print(console_colors.OKGREEN + f'Spam message where identified with the following accuracy: {accuracy}' + console_colors.ENDC)
    except:
        print(console_colors.FAIL + f'Something went wrong training the classifier!' + console_colors.ENDC)
    

