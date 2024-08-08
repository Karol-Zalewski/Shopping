import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4
FILE_DIRECTION = "shopping.csv"

def main():
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(FILE_DIRECTION)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Save the dictionary with the names of the months
    # tied to the number. In our data June is written Jun
    # which explains why we need to change it below
    months = {month : index for index, month in enumerate(calendar.month_abbr) if index}
    months["June"] = months.pop("Jun")

    # Evidence will contain all of our knowledge about the person
    # labels only the information if the person bought something
    evidence = []
    labels = []

    # We open the file and save the data in correct format
    with open(filename, 'r') as f:

        reader = csv.DictReader(f)
        for row in reader:

            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    # Return the tuple containing evidence and labels
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # We are using class KNeighborsClassifier from sklearn.neighbors
    model = KNeighborsClassifier(n_neighbors = 1)
    # We are training our model on the data we have
    model.fit(evidence, labels)

    # We return it
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_positive = 0
    guessed_right_positive = 0

    total_negative = 0
    guessed_right_negative = 0

    # We are looping through all of the labels from data and
    # our predictions from the model - which based on the evidence
    # gave us its prediction
    for label, prediction in zip(labels, predictions):

        # If the client bought something and we guessed right we
        # are adding 1 to both total_positive and guessed_right_positive
        # otherwise we are adding value only to total_positive
        if label == 1:
            total_positive += 1

            if label == prediction:
                guessed_right_positive += 1

        # If the client didn't buy something and we guessed right we
        # are adding 1 to both total_negative and guessed_right_negative
        # otherwise we are adding value only to total_negative
        elif label == 0:
            total_negative += 1

            if label == prediction:
                guessed_right_negative += 1
    
    # We are calculating both sensitivity and specificity based on
    # the number of times we got it right.
    sensitivity = guessed_right_positive / total_positive
    specificity = guessed_right_negative / total_negative

    return sensitivity, specificity



if __name__ == "__main__":
    main()
