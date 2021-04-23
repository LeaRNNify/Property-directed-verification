import time

import numpy as np

from dfa import DFA, complement
from modelPadding import RNNLanguageClasifier
from random_words import random_word
from teacher import Teacher


class PACTeacher(Teacher):

    def __init__(self, model, epsilon=0.001, delta=0.001):

        assert ((epsilon <= 1) & (delta <= 1))
        Teacher.__init__(self, model)
        self.epsilon = epsilon
        self.delta = delta
        self._log_delta = np.log(delta)
        self._log_one_minus_epsilon = np.log(1 - epsilon)
        self._num_equivalence_asked = 0

        self.prev_examples = {}

        self.is_counter_example_in_batches = isinstance(self.model, RNNLanguageClasifier)

        self.timeout = 600
        self.start_time = 0

    def equivalence_query(self, dfa: DFA):
        """
        Tests whether the dfa is equivalent to the model by testing random words.
        If not equivalent returns an example
        """

        number_of_rounds = int(
            (1 / self.epsilon) * (np.log(1 / self.delta) + np.log(2) * (self._num_equivalence_asked + 1)))

        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if self.is_counter_example_in_batches:
            if time.time() - self.start_time > self.timeout:
                return
            batch_size = 200
            for i in range(int(number_of_rounds / batch_size) + 1):
                batch = [random_word(self.model.alphabet) for _ in range(batch_size)]
                for x, y, w in zip(self.model.is_words_in_batch(batch), [dfa.is_word_in(w) for w in batch],
                                   batch):
                    if x != y:
                        return w
            return None

        else:
            for i in range(number_of_rounds):
                word = random_word(self.model.alphabet)
                if self.model.is_word_in(word) != dfa.is_word_in(word):
                    return word
            return None

    def model_subset_of_dfa_query(self, dfa: DFA):
        """
        Tests whether the model language is a subset of the dfa language by testing random words.
        If not subset returns an example
        """
        number_of_rounds = int(
            (1 / self.epsilon) * (np.log(1 / self.delta) + np.log(2) * (self._num_equivalence_asked + 1)))
        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if isinstance(self.model, RNNLanguageClasifier):
            batch_size = 200
            for i in range(int(number_of_rounds / batch_size) + 1):
                if time.time() - self.start_time > self.timeout:
                    return None

                batch = [random_word(self.model.alphabet) for _ in range(batch_size)]
                for x, y, w in zip(self.model.is_words_in_batch(batch), [dfa.is_word_in(w) for w in batch],
                                   batch):
                    if x and (not y):
                        return w
            return None

        else:
            for i in range(number_of_rounds):
                word = random_word(self.model.alphabet)
                if self.model.is_word_in(word) != dfa.is_word_in(word):
                    return word
            return None

    def model_subset_of_dfa_query_adv(self, dfa: DFA, start_time, timeout):
        """
        Tests whether the model language is a subset of the dfa language by testing random words.
        If not subset returns an example
        """

        number_of_rounds = int(
            (1 / self.epsilon) * (np.log(1 / self.delta) + np.log(2) * (self._num_equivalence_asked + 1)))
        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if isinstance(self.model, RNNLanguageClasifier):
            batch_size = 500
            for i in range(int(number_of_rounds / batch_size) + 1):
                if time.time() - start_time > timeout:
                    return None
                batch = []
                for _ in range(batch_size):
                    word = random_word(self.model.alphabet)
                    if len(word) < 30:
                        batch.append(word)
                batch = [random_word(self.model.alphabet) for _ in range(batch_size)]
                print(type(batch[50]))
                for x, y, w in zip(self.model.is_words_in_batch(batch) > 0.5, [dfa.is_word_in(w) for w in batch],
                                   batch):
                    if (x and (not y)) or ((not x) and y):
                        return w
            return None

    def membership_query(self, word):
        return self.model.is_word_in(word)

    def teach(self, learner, timeout=600):
        self.timeout = timeout
        self.start_time = time.time()
        self._num_equivalence_asked = 0
        learner.teacher = self
        i = 60
        while True:
            if time.time() - self.start_time > self.timeout:
                return
            if time.time() - self.start_time > i:
                i += 60
                print("AAMC - {} time has passed from starting AAMC, DFA is currently of size {}".format(
                    time.time() - self.start_time, len(learner.dfa.states)))

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                break
            num_of_ref = learner.new_counterexample(counter, self.is_counter_example_in_batches,
                                                    timeout=self.timeout - time.time() + self.start_time)
            self._num_equivalence_asked += num_of_ref

    def check_and_teach(self, learner, checker, timeout=600):
        self.timeout = timeout
        learner.teacher = self
        self._num_equivalence_asked = 0
        self.start_time = time.time()
        i = 60
        while True:
            if time.time() - self.start_time > self.timeout:
                return
            if time.time() - self.start_time > i:
                i += 60
                print("PDV - {} time has passed from starting PDV, DFA is currently of size {}".format(
                    time.time() - self.start_time, len(learner.dfa.states)))
            # Searching for counter examples in the spec:
            counter_example = checker.check_for_counterexample(learner.dfa)

            if counter_example is not None:
                if not self.model.is_word_in(counter_example):
                    self._num_equivalence_asked += 1
                    num = learner.new_counterexample(counter_example, self.is_counter_example_in_batches,
                                                     timeout=self.timeout - time.time() + self.start_time)
                    if num > 1:
                        self._num_equivalence_asked += num - 1
                else:
                    return counter_example

            # Searching for counter examples in the the model:
            else:

                counter_example = self.model_subset_of_dfa_query(learner.dfa)
                if counter_example is None:
                    return None
                else:
                    num_equivalence_used = learner.new_counterexample(counter_example,
                                                                      self.is_counter_example_in_batches)
                    if num_equivalence_used > 1:
                        self._num_equivalence_asked += num_equivalence_used - 1

    def adv_robustness(self, learner, neighbourhoodNFA, is_positive_word, timeout=600): 
        learner.teacher = self
        self._num_equivalence_asked = 0
        start_time = time.time()
        i = 5
        while True:
            if time.time() - start_time > timeout:
                print("Process TIMEOUT")
                return
            
            #if time.time() - start_time > i:
            #    i += 5
            #    print("PDV - {} time has passed from starting PDV, DFA is currently of size {}".format(
            #        time.time() - start_time, len(learner.dfa.states)))
            
            # Searching for adversarial example in the spec:
            if is_positive_word:
                print("Now, checking inclusion of neighbourhoodNFA of size %d and DFA of size %d"%(len(neighbourhoodNFA.states), len(learner.dfa.states)))
                adversarial_example = neighbourhoodNFA.inclusion(learner.dfa)
            else:
                print("Now, checking inclusion of neighbourhoodNFA of size %d and DFA of size %d"%(len(neighbourhoodNFA.states), len(learner.dfa.states)))
                adversarial_example = neighbourhoodNFA.inclusion(complement(learner.dfa))



            if adversarial_example is not None:
                
                if self.model.is_word_in(adversarial_example) != learner.dfa.is_word_in(adversarial_example):
                    print("******Adversarial Example found but RNN and DFA do not match******")
                    self._num_equivalence_asked += 1
                    num = learner.new_counterexample(adversarial_example, self.is_counter_example_in_batches)
                    if num > 1:
                        self._num_equivalence_asked += num - 1
                else:
                    print("******Adversarial Example of length %d found******"%(len(adversarial_example))) 
                    return adversarial_example

            # Searching for counter examples in the the model:
            else:
                print("--------Adversarial Example not found in this round--------")
                # Equivalence check
                counter_example = self.model_subset_of_dfa_query_adv(learner.dfa, start_time, timeout)
                print("counterexample: " + str(counter_example))
                if counter_example is None:
                    return None
                else:
                    num_equivalence_used = learner.new_counterexample(counter_example,
                                                                    self.is_counter_example_in_batches)
                    if num_equivalence_used > 1:
                        self._num_equivalence_asked += num_equivalence_used - 1
            print(self._num_equivalence_asked)


    
