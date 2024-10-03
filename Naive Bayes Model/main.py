import csv

# Open the csv file and iterate through it
trainingList = []
testingList = []

with open('Naive Bayes Model\SpamDetection.csv', 'r') as file:
  reader = csv.reader(file)

  # Split the data into testing and training sets
  i = 0
  for row in reader:
    if i <= 20:
      trainingList.append(row) # First 20 go into training
    else:
      testingList.append(row) # Last 10 go into testing
    i += 1
  # Remove first item from training list, as its just 'Target', 'Data'
  trainingList.pop(0)

print("\n1) Data \n")
# 1) Print the Training and Testing lists
print("Training Data:")
for item in trainingList:
  print(item)

print("\nTesting Data:")
for item in testingList:
  print(item)


# 2) Prior Probability
# P(Spam)
def PriorProbabilitySpam(data):
  spamCount = 0
  for item in data:
    if item[0] == 'spam':
      spamCount += 1
  return spamCount / len(data)

print("\n2) Prior Probabilites \n")
# Calculate and print
print("Prior Probability Spam: " + str(PriorProbabilitySpam(trainingList)))

# P(Ham)
def PriorProbabilityHam(data):
  hamCount = 0
  for item in data:
    if item[0] == 'ham':
      hamCount += 1
  return hamCount / len(data)

print("Prior Probability Ham: " + str(PriorProbabilityHam(trainingList)))

# Parse training list into unique words and counts
uniqueWords = []
spamCount = []
hamCount = []

# Iterate through the training data and add to unique words list and add the number of times it was seen for positive and negative
for item in trainingList:
  for word in item[1].split():
    if word.lower() not in uniqueWords:
      uniqueWords.append(word.lower())
      if item[0] == 'spam':
        spamCount.append(1)
        hamCount.append(0)
      else:
        spamCount.append(0)
        hamCount.append(1)
    else:
      index = uniqueWords.index(word.lower())
      if item[0] == 'spam':
        spamCount[index] += 1
      else:
        hamCount[index]+= 1


# ^^^^ change this above to split the words in to uniquePositive and uniqueNegative along with number of appearacnes in each
# so change how its counted in spam/ham count by not appending 0 to match the index of the uniqueword list
# we could work with the method above as it's still technically counting correctly but it might be easier to change

# Print the lists with the number of apperances in Spam and Ham
#for i in range(len(uniqueWords)):
  #print(uniqueWords[i] + ": [spam " + str(spamCount[i]) + "] " " / [ham " + str(hamCount[i]) + "]")

# 3) Conditional Probability P(Sentence|Spam) and P(Sentence|Ham)
def ConditionalProbabilitySpam(sentence, uniqueWords, spamCount, totalSpamWords):
  probability = 1
  for word in sentence.split():
    index = uniqueWords.index(word.lower())
    probability *= (spamCount[index]) / (totalSpamWords) # doesnt need Laplace smoothing because its given that its spam, so its gurenteed to not be 0
  return probability

def ConditionalProbabilityHam(sentence, uniqueWords, hamCount, totalHamWords):
  probability = 1
  for word in sentence.split():
    index = uniqueWords.index(word.lower())
    probability *= (hamCount[index]) / (totalHamWords) # doesnt need Laplace smoothing because its given that its ham, so its gurenteed to not be 0
  return probability

# Get total of ham words
totalHamWords = 0
for count in hamCount:
  totalHamWords += count

# Get total of spam words
totalSpamWords = 0
for count in spamCount:
  totalSpamWords += count

print("\n3) Conditional Probabilities \n")
# Calculate and print
for item in trainingList:
  if item[0] == 'spam':
    print("Conditional Probability Spam: " + str(ConditionalProbabilitySpam(item[1], uniqueWords, spamCount, totalSpamWords)))
  else:
    print("Conditional Probability Ham: " + str(ConditionalProbabilityHam(item[1], uniqueWords, hamCount, totalHamWords)))

# 4) Compute Posterior Probabilites for training set (probability of a sentence being spam or ham)
def PosteriorProbabilites(sentence, uniqueWords, spamCount, hamCount, totalSpamWords, totalHamWords):
  spamProbability = 1
  hamProbability = 1
  for word in sentence.split():
    index = uniqueWords.index(word.lower())
    spamProbability *= (spamCount[index] + 1) / (totalSpamWords + len(uniqueWords)) # add 1 to numerator and add num of total unique words to demnominator to make sure were not using 0
    hamProbability *= (hamCount[index] + 1) / (totalHamWords + len(uniqueWords)) # add 1 to numerator and add num of total unique words to demnominator to make sure were not using 0
  return spamProbability, hamProbability

print("\n4) Posterior Probabilites \n")
# Calculate and print
for item in trainingList:
  print("Posterior Probabilities (Spam, Ham): " + str(PosteriorProbabilites(item[1], uniqueWords, spamCount, hamCount, totalSpamWords, totalHamWords)))

# 5) Compute Posterior Probabilites for testing set and determine if spam or ham
def DetermineSpamOrHam(sentence, uniqueWords, spamCount, hamCount, totalSpamWords, totalHamWords):
  spamProbability = 1
  hamProbability = 1
  for word in sentence.split():
    if word.lower() in uniqueWords:
      index = uniqueWords.index(word.lower())
      spamProbability *= (spamCount[index] + 1) / (totalSpamWords + len(uniqueWords)) # add 1 to numerator and add num of total unique words to demnominator to make sure were not using 0
      hamProbability *= (hamCount[index] + 1) / (totalHamWords + len(uniqueWords)) # add 1 to numerator and add num of total unique words to demnominator to make sure were not using 0
  return spamProbability, hamProbability

print("\n5) Determine if spam or ham \n")
# Calculate and print
testSpamCount = 0
testHamCount = 0
for item in testingList:
  print(item[1]) 
  print("Posterior Probabilities (Spam, Ham): " + str(DetermineSpamOrHam(item[1], uniqueWords, spamCount, hamCount, totalSpamWords, totalHamWords)))
  spamProb, hamProb = DetermineSpamOrHam(item[1], uniqueWords, spamCount, hamCount, totalSpamWords, totalHamWords)
  if spamProb > hamProb:
    print("This is spam\n")
    testSpamCount += 1
  else:
    print("This is ham\n")
    testHamCount += 1

# 6) Accuracy of the spams and hams
print("\n6) Test set accuracy \n")
realSpamCount = 0
realHamCount = 0
for item in testingList:
  if item[0] == 'spam':
    realSpamCount += 1
  else:
    realHamCount += 1

print("Real spam count: " + str(realSpamCount))
print("Real ham count: " + str(realHamCount))
print("Test spam count: " + str(testSpamCount))
print("Test ham count: " + str(testHamCount))
if realSpamCount == testSpamCount:
  print("Test set accuracy: 100%")
else:
  print("Test set accuracy: " + str((len(testingList) - abs(realSpamCount - testSpamCount)) / len(testingList)))