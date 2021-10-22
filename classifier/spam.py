import pandas
import re
import numpy

LABEL_COLUMN = 'LABEL'
BODY_COLUMN = 'BODY'

class Classifier():

   def __init__(self, path):
      self.training_set = pandas.read_csv(path, sep='\t', header=None, names=[LABEL_COLUMN, BODY_COLUMN])
      self.vocabulary = []
      self.training_set_clean = None
      self._train()

   def _clean_data(self):
      # Removes punctuation
      self.training_set[BODY_COLUMN] = self.training_set[BODY_COLUMN].str.replace('\W', ' ', regex=True)

      # Removes multiple spaces with one.
      self.training_set[BODY_COLUMN] = self.training_set[BODY_COLUMN].str.replace('\s+', ' ', regex=True)

      # Lowercase everything
      self.training_set[BODY_COLUMN] = self.training_set[BODY_COLUMN].str.lower()
   

   def _build_vocabulary(self):
      self.training_set[BODY_COLUMN] = self.training_set[BODY_COLUMN].str.split()
      self.vocabulary = []
      for message in self.training_set[BODY_COLUMN]:
         for word in message:
            self.vocabulary.append(word)
      self.vocabulary = list(set(self.vocabulary))


   def _count_tokens(self):
      word_counts_per_message = {unique_word: [0] * len(self.training_set[BODY_COLUMN]) for unique_word in self.vocabulary}

      for index, message in enumerate(self.training_set[BODY_COLUMN]):
         for word in message:
            word_counts_per_message[word][index] += 1
            
      word_counts = pandas.DataFrame(word_counts_per_message)

      self.training_set_clean = pandas.concat([self.training_set, word_counts], axis=1)

   def _build_params(self):   
      # Isolating spam and ham messages first
      spam_messages = self.training_set_clean[self.training_set_clean[LABEL_COLUMN] == 1]
      ham_messages = self.training_set_clean[self.training_set_clean[LABEL_COLUMN] == 0]

      # P(Spam) and P(Ham)
      self.p_spam = len(spam_messages) / len(self.training_set_clean)
      self.p_ham = len(ham_messages) / len(self.training_set_clean)

      # N_Spam
      n_words_per_spam_message = spam_messages[BODY_COLUMN].apply(len)
      self.n_spam = n_words_per_spam_message.sum()

      # N_Ham
      n_words_per_ham_message = ham_messages[BODY_COLUMN].apply(len)
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
      self._clean_data()
      self._build_vocabulary()
      self._count_tokens()
      self._build_params()

   def classify(self, message):
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

   def classify_test_set(self, test_file_path):
      self.training_set = pandas.read_csv(test_file_path, sep='\t', header=None, names=[LABEL_COLUMN, BODY_COLUMN])
     
      accuracyCount = 0
      count = 0

      for index, row in self.training_set.iterrows():
         count += 1
         result = self.classify(row[BODY_COLUMN])
         if (row[LABEL_COLUMN] == 1 and result >= 0.5) or (row[LABEL_COLUMN] == 0 and result < 0.5):
            accuracyCount += 1

      return accuracyCount / count