from dfa import *
import copy
import random


class HammingNFA:
    """
    class for hammingNFA of a word for a given distance
    """

    def __init__(self, word, distance, alphabet=None):
        self.init_state = (0, 0)
        self.transitions = {}
        self.distance = distance

        self.word = word
        self.states = {
            (pos, err)
            for pos in range(len(self.word) + 1)
            for err in range(distance + 1)
        }
        # Remove states
        for err in range(1, distance + 1):
            for state in range(err):
                self.states.remove((state, err))
        self.final_states = [
            (pos, err) for (pos, err) in self.states if pos == len(self.word)
        ]
        if alphabet == None:
            self.alphabet = list(set(self.word))
        else:
            self.alphabet = alphabet

        word_length = distance - 1
        for err in range(distance, -1, -1):
            for pos in range(len(self.word), word_length, -1):
                tr = {letter: [] for letter in self.alphabet}

                for letter in self.alphabet:
                    if pos < len(self.word) and letter == self.word[pos]:
                        # No change
                        if pos < len(self.word):
                            tr[letter].append((pos + 1, err))
                        # Replacement
                    elif pos < len(self.word) and letter != self.word[pos]:
                        if pos < len(self.word) and err < distance:
                            tr[letter].append((pos + 1, err + 1))
                self.transitions[(pos, err)] = copy.deepcopy(dict(tr))
            word_length -= 1

        self.number_of_words = {state: 0 for state in self.states}

    def __str__(self):
        return str(self.init_state) + str(self.transitions) + str(self.final_states)

    def is_word_in(self, word):
        """
        returns whether a word belongs to the HammingNFA
        """
        current_states = {self.init_state}
        for letter in word:
            prev_states = current_states
            current_states = set()
            for state in prev_states:
                current_states.update(self.transitions[state][letter])

        if len(set(self.final_states).intersection(current_states)) != 0:
            return True
        else:
            return False

    # Can optimize this process
    def generate_all_accepting_words(self):
        """
        returns all words that are accepted by the HammingNFA
        """
        return self.generate_accepting_words(self.init_state)

    def generate_accepting_words(self, state):
        """
        returns all words that are accepted by a HammingNFA from a given state
        """

        all_words = []

        if state in self.final_states:
            all_words += [""]

        for letter in self.alphabet:
            successor_states = self.transitions[state][letter]

            # workaround for too many words
            if len(all_words) > 200:
                break

            for next_state in successor_states:
                all_words += [
                    letter + word for word in self.generate_accepting_words(next_state)
                ]

        return all_words

    def generate_num_accepting_words(self, length):
        """
        preprocessing step: returns the number of words that are accepted of a particular length
        """
        self.number_of_words = {
            (state, 0): int(state in self.final_states) for state in self.states
        }
        for i in range(1, length + 1):
            self.number_of_words.update({(state, i): 0 for state in self.states})
            for state in self.states:
                for letter in self.alphabet:
                    for next_state in self.transitions[state][letter]:
                        self.number_of_words[(state, i)] += self.number_of_words[
                            (next_state, i - 1)
                        ]

    def generate_random_word(self):
        """
        returns a random word that is accepted
        """
        random_length = len(
            self.word
        ) 
        return self.generate_random_word_length(random_length)

    # Algorithm taken from https://link.springer.com/article/10.1007/s00453-010-9446-5
    def generate_random_word_length(self, length):
        """
        returns a random word of a particular length that is accepted
        """
        rand_word = ""
        state = self.init_state
        for i in range(1, length + 1):
            transitions_list = []
            prob_list = []
            for letter in self.alphabet:
                for next_state in self.transitions[state][letter]:
                    transitions_list.append((letter, next_state))
                    prob_list.append(
                        self.number_of_words[(next_state, length - i)]
                        / self.number_of_words[(state, length - i + 1)]
                    )
            next_transition = random.choices(transitions_list, weights=prob_list)[0]
            state = next_transition[1]
            rand_word += next_transition[0]
        return rand_word

    def inclusion(self, dfa: DFA):
        """
        checks inclusion of the language of HammingNFA in the language of a given DFA
        """
        if set(self.alphabet) != set(dfa.alphabet):
            raise Exception("Alphabets do not match")

        stack = [(dfa.init_state, self.init_state, "")]
        visited = []

        while stack != []:

            (dfa_state, nfa_state, label) = stack.pop()
            if (dfa_state, nfa_state) in visited:
                continue
            else:
                visited.append((dfa_state, nfa_state))
            for letter in self.alphabet:

                next_dfa_state = dfa.transitions[dfa_state][letter]
                next_states = []
                for next_nfa_state in self.transitions[nfa_state][letter]:

                    if (
                        next_nfa_state in self.final_states
                        and next_dfa_state not in dfa.final_states
                    ):
                        return tuple(label + letter)
                    next_states.append((next_dfa_state, next_nfa_state, label + letter))

                stack = next_states + stack

        return None

    def distance_random_word(self, og_word, distance, og_alphabet) -> str:
        """
        return a random word within certain levenstein distance
        """
        word = og_word
        for _ in range(distance):

            pos = random.randint(0, len(og_word) - 1)
            random_letter = random.choice(og_alphabet)
            word = word[:pos] + random_letter + word[pos + 1 :]

        return word

