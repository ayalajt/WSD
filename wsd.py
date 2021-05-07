#############################################################################################################################
# File:        wsd.py
# Author:      Jesus Ayala
# Date:        03/30/2021
# Class:       CMSC 416
# Description: A WSD using a Decision List classifier to tag senses to a text file given a training file 
#############################################################################################################################

# Word Sense Disambiguation refers to providing a sense to a word that is ambigious in a given context. This can be done by using a training file to "train" a model which
# can then be used on a test file to assign senses to ambigious words. These words can be disambiguated using a number of different methods, but for our purpose a decision 
# list classifier is used. It takes certain features of the training file and applies them to the test file by using if-else statements to see if a feature can be used or not.
# For example, if the word before "line" (a word that is ambigious) is commonly "telephone", then you can create a feature test with this by checking if "line" is preceeded by
# "telephone" in the test file. If it is, apply the phone sense, and if it is not, then continue to other features. My train of thought was to first simply go through the training file, grabbing
# all of the data needed while creating 6 features from Yarowsky's paper. I started with the first one, which checked the word to the right of the ambigious word, and I grabbed the most common ones to check 
# with the test file. After this was finished, I continued to the next feature, which checked the word to the left of the ambigious word. This was repeated for 6 of Yarowsky's features,
# and I also implemented a feature that finds the most common words in each sense's sentences and checks if they exist in the test file in order to apply a sense tag. The baseline accuracy 
# was calculated by simply applying the most frequent sense in the training file, and this resulted in a baseline accuracy of 42.86%. The following features were then implemented: 
#
# Feature 1: Store the most common words to the right of the ambigious word. Create two lists for these common words - one for phone sense and one for product sense. Check the words to the
# right of the ambigious word in the test file. This led to an accuracy score increase to 53.17%.
# Feature 2: Store the most common words to the left of the ambigious word. Create two lists for these common words - one for phone sense and one for product sense. Check the words to the left
# of the ambigious word in the test file. This led to an enormous accuracy score increase to 82.54%.
# Feature 3: Store the K number of words to the right of the ambigious word. Create two lists for these words - one for phone sense and one for product sense. Check the K number of words
# to the right of the ambigious word in the test file. This did not end up providing a difference, as the accuracy score remained at 82.54%.
# Feature 4: Store the 2 words to the left of the ambigious word. Grab the 2 words to the left of the ambigious word in the test file and compare them to the lists created. Once again, this did not
# improve or lower the accuracy score, as it remained at 82.54%.
# Feature 5: Store the word to the left and to the right of the ambigious word. In the test file, grab the left and right words as well and check if they are in either of the phone sense list or
# product sense list. This did not change my accuracy either, as it remained at 82.54%
# Feature 6: Store the 2 words to the right of the ambigious word in the training file. Then grab the 2 words to the right of the ambigious word in the training file, and compare it to the phone and product
# lists. This led to a small increase as the accuracy score improved to 83.33%.
# Feature 7: Originally, I found the 5 most common words in each sense's word list and stored them in the two sense's lists. In the test file, if the word exists in one of the lists, apply the respective sense to the 
# line. This led to a score decrease to 80.17% unfortunately, but after configuring the list of the most common words for each sense, I found that grabbing the 2 most common words for each sense led to a significant 
# increase to 87.3%.
#
# After every sentence in the test file has been given a sense tag, each feature's specifics are printed out to the third command line argument file name. For example, it will print "If line is followed by "and", then 
# the word sense is phone." It follows this format for every word used in the features, and below every feature, the log-likelihood score is printed.
# For example input, the file can be run from the command line and it needs three text files: the training file, testing file, and the output file. It can then be printed to another file with
# the command: "python wsd.py line-train.txt line-test.txt my-model.txt > output.txt". The output will then be printed onto the output text file, where each line follows the line-key's fanswer format by printing the line
# information and the sense tag.

import sys
import re
import collections
from collections import Counter
import math

def main():

    # The first part of this program grabs the name of the three files to be used
    fileForTraining = -1
    fileForSenseTag = -1
    try:
        fileForTraining = sys.argv[1]
        fileForSenseTag = sys.argv[2]
        fileForModel = sys.argv[3]
    except:
        print("More arguments required")
        quit()

    openTrainingFile = open(fileForTraining, encoding = "utf8")
    line = ""
    word = ""


    # a list of words to be ignored when finding the most common words in each sense's wordset
    stopList = ["the", "to", "a", "of", "and", ".", ",","in","lines","line","for","The","\"","that","is","was","by","from","with","as","he","said","or","be","have","it",
                "which","his","its","on","has","about","up","at","an","are"]
    # Go through each line in the training file, ignore brackets

    # For every feature, initialize the lists to be used for the two senses
    # Feature 1 
    wordsToRightPhone = []
    wordsToRightProduct = []
    # Feature 2
    wordsToLeftPhone = []
    wordsToLeftProduct = []
    # Feature 3
    K = 3
    KwordsPhone = []
    KwordsProduct = []
    # Feature 4
    multiWordsToLeftPhone = []
    multiWordsToLeftProduct = []
    # Feature 5
    surrWordsPhone = []
    surrWordsProduct = []
    # Feature 6
    multiWordsToRightPhone = []
    multiWordsToRightProduct = []

    # Keep track of the number of senses in the training file
    phoneSenseTotal = 0
    productSenseTotal = 0

    # Lists used to store every word in the sense's sentences
    sensePhoneWords = []
    senseProductWords = []
    
    for line in openTrainingFile:
        if "senseid=\"phone\"" in line:
            phoneSenseTotal = phoneSenseTotal + 1

            # Skip the context tag
            next(openTrainingFile)
            sentence = next(openTrainingFile)

            # Clean up sentences
            sentence = sentence.replace("<s>","")
            sentence = sentence.replace("</s>","")
            sentence = sentence.replace("<@>","")
            sentence = sentence.replace("<head>","")
            sentence = sentence.replace("</head>","")
            sentence = sentence.replace("<p>","")
            sentence = sentence.replace("</p>","")

            sentence = ' '.join(sentence.split())
            listOfWords = sentence.split(" ")

            # Loop through the current sentence
            for wordNum in range(len(listOfWords)):
                currentWord = listOfWords[wordNum]

                # If the current word is the ambigious word, use it to grab each feature's dataset
                if currentWord == "line" or currentWord == "lines":
                    prevWord = listOfWords[wordNum-1]
                    prevPrevWord = listOfWords[wordNum-2]
                    nextWord = listOfWords[wordNum+1]
                    if wordNum + 2 < len(listOfWords):
                        nextNextWord = listOfWords[wordNum+2]

                    # For the k number feature, grab the number of words from 0 to K, making sure not to 
                    # go out of the sentence's bounds
                    KWords = ""
                    for grabWord in range(0, K):
                        if wordNum + 1 + grabWord < len(listOfWords):
                            KWords = KWords + " " + listOfWords[(wordNum+1)+grabWord]
                    KWords = ' '.join(KWords.split())
                    KWords = KWords.split(" ")

                    # Make sure the number of k words is of k length
                    if len(KWords) == K:
                        KwordsPhone.append(KWords)

                    # Store all of the necessary data in the respective phone sense's lists
                    wordsToRightPhone.append(nextWord)
                    wordsToLeftPhone.append(prevWord)
                    multiWordsToLeftPhone.append((prevPrevWord, prevWord))
                    multiWordsToRightPhone.append((nextWord, nextNextWord))
                    surrWordsPhone.append((prevWord, nextWord))

                # Check if the word is in the stop list. If it is, don't add it to the phone's wordset.
                # If it isn't, add it
                inStopList = False
                for element in stopList:
                    if currentWord == element:
                        inStopList = True
                if inStopList == False:
                    sensePhoneWords.append(currentWord)
            
        elif "senseid=\"product\"" in line:
            productSenseTotal = productSenseTotal + 1
            
            # Skip the context tag
            next(openTrainingFile)
            sentence = next(openTrainingFile)
            
            # Clean up sentences
            sentence = sentence.replace("<s>","")
            sentence = sentence.replace("</s>","")
            sentence = sentence.replace("<@>","")
            sentence = sentence.replace("<head>","")
            sentence = sentence.replace("</head>","")
            sentence = sentence.replace("<p>","")
            sentence = sentence.replace("</p>","")

            sentence = ' '.join(sentence.split())
            listOfWords = sentence.split(" ")
            for wordNum in range(len(listOfWords)):
                currentWord = listOfWords[wordNum]
                
                # If the current word is the ambigious word, use it to grab each feature's dataset
                if currentWord == "line" or currentWord == "lines":
                    prevWord = listOfWords[wordNum-1]
                    prevPrevWord = listOfWords[wordNum-2]
                    nextWord = listOfWords[wordNum+1]
                    if wordNum + 2 < len(listOfWords):
                        nextNextWord = listOfWords[wordNum+2]
                    
                    # For the k number feature, grab the number of words from 0 to K, making sure not to 
                    # go out of the sentence's bounds
                    KWords = ""
                    for grabWord in range(0, K):
                        if wordNum+1+grabWord < len(listOfWords):
                            KWords = KWords + " " + listOfWords[(wordNum+1)+grabWord]
                    KWords = ' '.join(KWords.split())
                    KWords = KWords.split(" ")
                    if len(KWords) == K:
                        KwordsProduct.append(KWords)
                    
                    # Store all of the necessary data in the respective product sense's lists
                    wordsToRightProduct.append(nextWord)
                    wordsToLeftProduct.append(prevWord)
                    multiWordsToLeftProduct.append((prevPrevWord, prevWord))
                    multiWordsToRightProduct.append((nextWord, nextNextWord))
                    surrWordsProduct.append((prevWord, nextWord))

                # Check if the word is in the stop list. If it is, don't add it to the phone's wordset.
                # If it isn't, add it
                inStopList = False
                for element in stopList:
                    if currentWord == element:
                        inStopList = True
                if inStopList == False:
                    senseProductWords.append(currentWord)
    
    # To find the most frequent sense tag, compare the number of phone sense sentences to product sense sentences
    majority = "none"
    if (phoneSenseTotal > productSenseTotal):
        majority = "phone"
    else:
        majority = "product"

    # For every feature, grab the N most common words for each list, where N is tweaked for best results
    # Feature 1
    myCounter = Counter(wordsToRightPhone)
    mostFreqNextPhoneWords = myCounter.most_common(5)
    myCounter = Counter(wordsToRightProduct)
    mostFreqNextProductWords = myCounter.most_common(7)

    # Feature 2
    myCounter = Counter(wordsToLeftProduct)
    mostFreqPrevProductWords = myCounter.most_common(4)
    myCounter = Counter(wordsToLeftPhone)
    mostFreqPrevPhoneWords = myCounter.most_common(3)

    # Feature 3
    tupleKWords = tuple(tuple(sub) for sub in KwordsPhone)
    myCounter = Counter(tupleKWords)
    mostFreqPhoneKWords = myCounter.most_common(3)
    tupleKWords = tuple(tuple(sub) for sub in KwordsProduct)
    myCounter = Counter(tupleKWords)
    mostFreqProductKWords = myCounter.most_common(3)

    # Feature 4
    myCounter = Counter(multiWordsToLeftPhone)
    multiWordsToLeftPhoneFreq = myCounter.most_common(3)
    myCounter = Counter(multiWordsToLeftProduct)
    multiWordsToLeftProductFreq = myCounter.most_common(3)

    # Feature 5
    myCounter = Counter(surrWordsPhone)
    surrWordsPhoneFreq = myCounter.most_common(3)
    myCounter = Counter(surrWordsProduct)
    surrWordsProductFreq = myCounter.most_common(3)

    # Feature 6
    myCounter = Counter(multiWordsToRightPhone)
    multiWordsToRightPhoneFreq = myCounter.most_common(3)
    myCounter = Counter(multiWordsToRightProduct)
    multiWordsToRightProductFreq = myCounter.most_common(3)

    # Feature 7
    myCounter = Counter(sensePhoneWords)
    commonPhoneWordsFreq = myCounter.most_common(2)
    myCounter = Counter(senseProductWords)
    commonProductWordsFreq = myCounter.most_common(2)

    # For every feature, add the most common words to a new list that ignores the count
    # Feature 1
    rightWordsPhoneFeatures = []
    for (word, count) in mostFreqNextPhoneWords:
        if "." != word and "," != word:
            rightWordsPhoneFeatures.append(word)
    rightWordsProductFeatures = []
    for (word, count) in mostFreqNextProductWords:
        if "." != word and "," != word and (word not in rightWordsPhoneFeatures):
            rightWordsProductFeatures.append(word)

    # Feature 2
    leftWordsPhoneFeatures = []
    for (word, count) in mostFreqPrevPhoneWords:
        leftWordsPhoneFeatures.append(word)
    leftWordsProductFeatures = []
    for (word, count) in mostFreqPrevProductWords:
        if word not in leftWordsPhoneFeatures:
            leftWordsProductFeatures.append(word)

    # Feature 3
    KwordsPhoneFeatures = []
    for (words, count) in mostFreqPhoneKWords:
            KwordsPhoneFeatures.append(words)
    KwordsProductFeatures = []
    for (words, count) in mostFreqProductKWords:
            KwordsProductFeatures.append(words)
            
    # Feature 4
    multiWordsToLeftPhoneFeatures = []
    for ((firstWord, secondWord), count) in multiWordsToLeftPhoneFreq:
        multiWordsToLeftPhoneFeatures.append((firstWord, secondWord))
    multiWordsToLeftProductFeatures = []
    for ((firstWord, secondWord), count) in multiWordsToLeftProductFreq:
        multiWordsToLeftProductFeatures.append((firstWord, secondWord))

    # Feature 5
    surrWordsPhoneFeatures = []
    for (words, count) in surrWordsPhoneFreq:
        surrWordsPhoneFeatures.append(words)
    surrWordsProductFeatures = []
    for (words, count) in surrWordsProductFreq:
        surrWordsProductFeatures.append(words)

    # Feature 6
    multiWordsToRightPhoneFeatures = []
    for ((firstWord, secondWord), count) in multiWordsToRightPhoneFreq:
        multiWordsToRightPhoneFeatures.append((firstWord, secondWord))
    multiWordsToRightProductFeatures = []
    for ((firstWord, secondWord), count) in multiWordsToRightProductFreq:
        multiWordsToRightProductFeatures.append((firstWord, secondWord))

    # Feature 7
    commonPhoneWordsFeatures = []
    for (word, count) in commonPhoneWordsFreq:
        commonPhoneWordsFeatures.append(word)
    commonProductWordsFeatures = []
    for (word, count) in commonProductWordsFreq:
        commonProductWordsFeatures.append(word)

    openFileToSenseTag = open(fileForSenseTag, encoding = "utf8")

    # initialize all values required for the log likelihood
    featureOnePhone = 0
    featureOneProduct = 0
    featureOneTotal = 0

    featureTwoPhone = 0
    featureTwoProduct = 0
    featureTwoTotal = 0

    featureThreePhone = 0
    featureThreeProduct = 0
    featureThreeTotal = 0

    featureFourPhone = 0
    featureFourProduct = 0
    featureFourTotal = 0

    featureFivePhone = 0
    featureFiveProduct = 0
    featureFiveTotal = 0

    featureSixPhone = 0
    featureSixProduct = 0
    featureSixTotal = 0

    featureSevenPhone = 0
    featureSevenProduct = 0
    featureSevenTotal = 0
    
    # To get the baseline, just change this to false so the features are not used and every word sense is assigned the most frequent sense
    useFeatures = True

    for line in openFileToSenseTag:
        # in the test text file, find every instance id and adjust it to the format in the key
        if "instance id=" in line:
            sense = "none"
            lineID = re.findall('"([^"]*)"', line)
            answerLine = "<answer instance=\"" + str(lineID[0]) + "\" senseid="

            # Skip the context tag
            next(openFileToSenseTag)
            sentence = next(openFileToSenseTag)

            # Clean up sentences
            sentence = sentence.replace("<s>","")
            sentence = sentence.replace("</s>","")
            sentence = sentence.replace("<@>","")
            sentence = sentence.replace("<head>","")
            sentence = sentence.replace("</head>","")
            sentence = sentence.replace("<p>","")
            sentence = sentence.replace("</p>","")
            sentence = ' '.join(sentence.split())
            listOfWords = sentence.split(" ")
            for wordNum in range(len(listOfWords)):
                currentWord = listOfWords[wordNum]
                if currentWord == "line" or currentWord == "lines":

                    # Grab the necessary data to be used to check every feature
                    prevWord = listOfWords[wordNum-1]
                    prevPrevWord = listOfWords[wordNum-2]
                    nextWord = listOfWords[wordNum+1]
                    if wordNum + 2 < len(listOfWords):
                        nextNextWord = listOfWords[wordNum+2]

                    # The way the tests are set up is that it goes down the list and so it can be overwritten, 
                    # this led to a higher accuracy score than not changing the sense after it has been assigned.
                    # Log information is also kept track of
                    if useFeatures == True:

                        # Feature 1 check
                        if nextWord in rightWordsPhoneFeatures:
                            sense = "phone"
                            featureOnePhone = featureOnePhone + 1
                            featureOneTotal += 1
                        elif nextWord in rightWordsProductFeatures:
                            sense = "product"
                            featureOneProduct = featureTwoProduct + 1
                            featureOneTotal += 1

                        # Feature 2 check
                        if prevWord in leftWordsPhoneFeatures:
                            sense = "phone"
                            featureTwoPhone = featureTwoPhone + 1
                            featureTwoTotal += 1
                        elif prevWord in leftWordsProductFeatures:
                            sense = "product"
                            featureTwoProduct = featureTwoProduct + 1
                            featureTwoTotal += 1

                        # Feature 3 check
                        KWords = ""
                        for grabWord in range(0, K):
                            if wordNum+1+grabWord < len(listOfWords):
                                KWords = KWords + " " + listOfWords[(wordNum+1)+grabWord]
                        KWords = ' '.join(KWords.split())
                        KWords = KWords.split(" ")
                        tupleKWords = tuple(KWords)
                        if tupleKWords in KwordsPhoneFeatures:
                            sense = "phone"
                            featureThreePhone = featureThreePhone + 1
                            featureThreeTotal += 1
                        elif tupleKWords in KwordsProductFeatures:
                            sense = "product"
                            featureThreeProduct = featureThreeProduct + 1            
                            featureThreeTotal += 1

                        # Feature 4 check
                        if (prevPrevWord, prevWord) in multiWordsToLeftPhoneFeatures:
                            sense = "phone"
                            featureFourPhone = featureFourPhone + 1
                            featureFourTotal += 1
                        elif (prevPrevWord, prevWord) in multiWordsToLeftProductFeatures:
                            sense = "product"
                            featureFourProduct = featureFourProduct + 1
                            featureFourTotal += 1

                        # Feature 5 check
                        if (prevWord, nextWord) in surrWordsPhoneFeatures:
                            sense = "phone"
                            featureFivePhone = featureFivePhone + 1
                            featureFiveTotal += 1
                        elif (prevWord, nextWord) in surrWordsProductFeatures:
                            sense = "product"
                            featureFiveProduct = featureFiveProduct + 1
                            featureFiveTotal += 1

                        # Feature 6 check
                        if (nextWord, nextNextWord) in multiWordsToRightPhoneFeatures:
                            sense = "phone"
                            featureSixPhone = featureSixPhone + 1
                            featureSixTotal += 1
                        elif (nextWord, nextNextWord) in multiWordsToRightProductFeatures:
                            sense = "product"
                            featureSixProduct = featureSixProduct + 1
                            featureSixTotal += 1
                if useFeatures == True:
                    # Feature 7 check
                    if sense == "none":
                        for word in listOfWords:
                            if word in commonPhoneWordsFeatures:
                                sense = "phone"
                                featureSevenPhone = featureSevenPhone + 1
                                featureSevenTotal += 1
                            elif word in commonProductWordsFeatures:
                                sense = "product"
                                featureSevenProduct = featureSevenProduct + 1
                                featureSevenTotal += 1

                # Last case aka the default case: If it reaches here and the sense has not been assigned,
                # then assign it the most frequent sense tag
                if sense == "none":
                    sense = majority

            answerLine = answerLine + "\"" + str(sense) + "\"/>" 
            print(answerLine)

    if useFeatures == True:

        # Calculate the log-likelihood for every feature

        featureOnePhoneProb = featureOnePhone / featureOneTotal
        featureOneProductProb = featureOneProduct / featureOneTotal
        featureOneLog = math.log10(featureOnePhoneProb / featureOneProductProb)

        featureTwoPhoneProb = featureTwoPhone / featureTwoTotal
        featureTwoProductProb = featureTwoProduct / featureTwoTotal
        featureTwoLog = math.log10(featureTwoPhoneProb / featureTwoProductProb)

        featureThreePhoneProb = featureThreePhone / featureThreeTotal
        featureThreeProductProb = featureThreeProduct / featureThreeTotal
        featureThreeLog = math.log10(featureThreePhoneProb / featureThreeProductProb)

        featureFourPhoneProb = featureFourPhone / featureFourTotal
        featureFourProductProb = featureFourProduct / featureFourTotal
        featureFourLog = math.log10(featureFourPhoneProb / featureFourProductProb)

        featureFivePhoneProb = featureFivePhone / featureFiveTotal
        featureFiveProductProb = featureFiveProduct / featureFiveTotal
        featureFiveLog = math.log10(featureFivePhoneProb / featureFiveProductProb)

        featureSixPhoneProb = featureSixPhone / featureSixTotal
        featureSixProductProb = featureSixProduct / featureSixTotal
        featureSixLog = math.log10(featureSixPhoneProb / featureSixProductProb)

        featureSevenPhoneProb = featureSevenPhone / featureSevenTotal
        featureSevenProductProb = featureSevenProduct / featureSevenTotal
        featureSevenLog = math.log10(featureSevenPhoneProb / featureSevenProductProb)

    modelFile = open(fileForModel, "w", encoding= "utf8")

    if useFeatures == True:
        # print every feature's word check as well as the log-likelihood to the specified text file
        
        # Feature 1 
        for word in rightWordsPhoneFeatures:
            newLine = "If line is followed by \"" + str(word) + "\", then the word sense is phone.\n"
            modelFile.write(newLine)
        for word in rightWordsProductFeatures:
            newLine = "If line is followed by \"" + str(word) + "\", then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureOneLog) + "\n")
        
        # Feature 2
        for word in leftWordsPhoneFeatures:
            newLine = "If line is preceeded by \"" + str(word) + "\", then the word sense is phone.\n"
            modelFile.write(newLine)
        for word in leftWordsProductFeatures:
            newLine = "If line is preceeded by \"" + str(word) + "\", then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureTwoLog) + "\n")

        # Feature 3
        for words in KwordsPhoneFeatures:
            newLine = "If line is followed by the k=" + str(K) + " words \"" + str(words)[1:-1] + "\", then the word sense is phone.\n"
            modelFile.write(newLine)
        for words in KwordsProductFeatures:
            newLine = "If line is followed by the k=" + str(K) + " words \"" + str(words)[1:-1] + "\", then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureThreeLog) + "\n")

        # Feature 4
        for words in multiWordsToLeftPhoneFeatures:
            newLine = "If line is preceeded by the two words \"" + words[0] + " " + words[1] + "\", then the word sense is phone.\n"
            modelFile.write(newLine)
        for words in multiWordsToLeftProductFeatures:
            newLine = "If line is preceeded by the two words \"" + words[0] + " " + words[1] + "\", then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureFourLog) + "\n")

        # Feature 5
        for words in surrWordsPhoneFeatures:
            newLine = "If line is preceeded by \"" + words[0] + "\" and then followed by \"" + words[1] + "\", then the word sense is phone.\n"
            modelFile.write(newLine)
        for words in surrWordsProductFeatures:
            newLine = "If line is preceeded by \"" + words[0] + "\" and then followed by \"" + words[1] + "\", then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureFiveLog) + "\n")

        # Feature 6
        for words in multiWordsToRightPhoneFeatures:
            newLine = "If line is followed by the two words \"" + words[0] + " " + words[1] + "\", then the word sense is phone.\n"
            modelFile.write(newLine)
        for words in multiWordsToRightProductFeatures:
            newLine = "If line is followed by the two words \"" + words[0] + " " + words[1] + "\", then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureSixLog) + "\n")

        # Feature 7
        for word in commonPhoneWordsFeatures:
            newLine = "If line contains the word \"" + word + "\" in its sentence, then the word sense is phone.\n"
            modelFile.write(newLine)
        for word in commonProductWordsFeatures:
            newLine = "If line contains the word \"" + word + "\" in its sentence, then the word sense is product.\n"
            modelFile.write(newLine)
        modelFile.write("Log-likelihood score = " + str(featureSevenLog) + "\n")

if __name__ == '__main__':
    main()