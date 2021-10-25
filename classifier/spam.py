import pandas
import re

class Classifier():
   """
   Represents an object which takes in a training set of messages
   and uses the Naive Bayes to classify new messages.
   """   

   def __init__(self, path, label_column = "LABEL", body_column = "BODY", start_row = 0, seperator = "\t"):
      self.body_column = body_column
      self.label_column = label_column

      self.training_set = pandas.read_csv(path, sep=seperator, header=start_row, names=[self.label_column, self.body_column])
      self.vocabulary = []
      self.training_set_clean = None
      self._train()

   def _clean_data(self):
      """
      Takes in a training set and cleans it up by removing punctuation, removing multiple spaces, 
      and converting to lowercase.
      """      
      
      # Removes punctuation
      self.training_set[self.body_column] = self.training_set[self.body_column].str.replace('\W', ' ', regex=True)

      # Removes multiple spaces with one.
      self.training_set[self.body_column] = self.training_set[self.body_column].str.replace('\s+', ' ', regex=True)

      # Lowercase everything
      self.training_set[self.body_column] = self.training_set[self.body_column].str.lower()
   

   def _build_vocabulary(self):
      """Builds a unique set of words from the training set.
      """

      self.training_set[self.body_column] = self.training_set[self.body_column].str.split()
      self.vocabulary = []
      for message in self.training_set[self.body_column]:
         for word in message:
            self.vocabulary.append(word)
      self.vocabulary = list(set(self.vocabulary))


   def _count_tokens(self):
      """Computes the count of all tokens in each message.
      """

      word_counts_per_message = {unique_word: [0] * len(self.training_set[self.body_column]) for unique_word in self.vocabulary}

      for index, message in enumerate(self.training_set[self.body_column]):
         for word in message:
            word_counts_per_message[word][index] += 1
            
      word_counts = pandas.DataFrame(word_counts_per_message)

      self.training_set_clean = pandas.concat([self.training_set, word_counts], axis=1)

   def _build_params(self):
      """Builds all parameters needed for the classifier to classify a message.
      """      

      # Isolating spam and ham messages first
      spam_messages = self.training_set_clean[self.training_set_clean[self.label_column] == 1]
      ham_messages = self.training_set_clean[self.training_set_clean[self.label_column] == 0]

      # P(Spam) and P(Ham)
      self.p_spam = len(spam_messages) / len(self.training_set_clean)
      self.p_ham = len(ham_messages) / len(self.training_set_clean)

      # N_Spam
      n_words_per_spam_message = spam_messages[self.body_column].apply(len)
      self.n_spam = n_words_per_spam_message.sum()

      # N_Ham
      n_words_per_ham_message = ham_messages[self.body_column].apply(len)
      self.n_ham = n_words_per_ham_message.sum()

      # Initiate parameters
      self.parameters_spam = {unique_word:0 for unique_word in self.vocabulary}
      self.parameters_ham = {unique_word:0 for unique_word in self.vocabulary}

      # Calculate parameters
      for word in self.vocabulary:
         n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
         p_word_given_spam = (n_word_given_spam + 1) / (self.n_spam + 2)
         self.parameters_spam[word] = p_word_given_spam

         n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
         p_word_given_ham = (n_word_given_ham + 1) / (self.n_ham + 1)
         self.parameters_ham[word] = p_word_given_ham

   def _train(self):
      """Trains the classifier.
      """

      self._clean_data()
      self._build_vocabulary()
      self._count_tokens()
      self._build_params()

   def classify(self, message):
      """
      Classifies a single message and returns the probability of being spam.
      If the result is greater than or equal to 0.5, it is spam. Otherwise, it is ham/

      Args:
          message (str): A message to classify.

      Returns:
          double: The probability of being spam.
      """
      
      message = re.sub('\W', ' ', message)
      message = message.lower().split()

      p_spam_given_message = self.p_spam
      p_ham_given_message = self.p_ham

      for word in message:
         if word in self.parameters_spam:
            p_spam_given_message *= self.parameters_spam[word]

         if word in self.parameters_ham: 
            p_ham_given_message *= self.parameters_ham[word]

      return p_spam_given_message / (p_spam_given_message + p_ham_given_message)

   def classify_test_set(self, test_file_path, label_column = "LABEL", body_column = "BODY", start_row = 0, seperator = "\t"):
      """Classifies all messages in the test file path and computes the accuracy of the classifier.

      Args:
          test_file_path (str): The path of the test file.

      Returns:
          double: The accuracy of the classifier.
      """      

      self.training_set = pandas.read_csv(test_file_path, sep=seperator, header=start_row, names=[label_column, body_column])
     
      accuracyCount = 0
      count = 0

      for index, row in self.training_set.iterrows():
         count += 1
         result = self.classify(row[self.body_column])
         if (row[label_column] == 1 and result >= 0.5) or (row[label_column] == 0 and result < 0.5):
            accuracyCount += 1

      return accuracyCount / count