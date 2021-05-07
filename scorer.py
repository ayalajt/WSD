#############################################################################################################################
# File:        scorer.py
# Author:      Jesus Ayala
# Date:        03/30/2021
# Class:       CMSC 416
# Description: A scoring file from a WSD classifier; it takes two files and compares the accuracy between the two
#############################################################################################################################

# The scorer file compares two files, a file that is tagged using a Decision list classifier, and a file that is the sense file key, and compares the
# similarity between the two's sense tags. This results in an accuracy score being calculated as well as a confusion matrix. This is done by going 
# line by line in each file, grabbing the sense tag, and comparing the tags between the line in both files. If they are the same, then it was tagged correctly. 
# Then divide the number of correctly tagged words by the total number of sentences, which results in the accuracy score being printed. For the confusion matrix, 
# pandas was used by running "pip install pandas" in the command line. It is then passed a list containing all the training file sense tags as well as a row that 
# contains all the key file sense tags. It then uses these together to create the confusion matrix. An example input would be 
# "python scorer.py output.txt line-key.txt > matrix.txt", and the output will contain the accuracy score at the top and the confusion matrix down below.

import sys
import pandas as pd
import re

def main():

    # First grab the file names
    fileTagged = -1
    fileTaggedKey = -1
    try:
        fileTagged = sys.argv[1]
        fileTaggedKey = sys.argv[2]
    except:
        print("More arguments required")
        quit()

    myResults = []
    keyResults = []

    # For the sense tagged file, go through the file and grab every sense tag for every line
    openTaggedFile = open(fileTagged, encoding = "utf8")
    for line in openTaggedFile:
        grabSense = re.findall('"([^"]*)"', line)
        myResults.append(grabSense[1])

    # For the sense tag key file, go through the file and grab every sense tag for every line
    openKeyFile = open(fileTaggedKey, encoding = "utf8")
    for line in openKeyFile:
        grabSense = re.findall('"([^"]*)"', line)
        keyResults.append(grabSense[1])
    
    sameTags = 0
    tagTotal = len(myResults)

    # iterate through the number of total tags (which would be the same total for both files) and compare the tags at the same index.
    # if they are the same, then it was tagged correctly, so increment the same tags value
    for i in range(tagTotal):
        if myResults[i] == keyResults[i]:
            sameTags = sameTags + 1
    
    # Calculate the accuracy by dividing the number of correct tags by the total tags, multiplying by 100, and then rounding to 2 significant digits
    accuracy = sameTags / tagTotal
    accuracy = accuracy * 100
    accuracy = round(accuracy, 2)
    print("Overall Accuracy: " + str(accuracy) + "%")
    print()

    # Here, pandas is used for the confusion matrix. The first four lines are formatting options, and then actual contains the row with the key file's sense tags
    # and predicted contains the trained file's sense tags. Create a matrix using both of these and print the results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    actual = pd.Series(keyResults, name="Actual")
    predicted = pd.Series(myResults, name="Predicted")
    matrix = pd.crosstab(actual, predicted)
    print(matrix)
            
if __name__ == '__main__':
    main()