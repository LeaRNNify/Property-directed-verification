from dfa import *
import copy
import random


# Based on the method presented in Session 4 of https://link.springer.com/article/10.1007/s10032-002-0082-8
class LevensteinNFA:
    """
    class for levensteinNFA of a word for a given distance
    """

    def __init__(self, word_or_dfa, distance, alphabet=None):

        self.transitions = {}
        self.distance = distance

        # For word case
        if isinstance(word_or_dfa, str):
            self.init_state = (0, 0)
            self.word = word_or_dfa
            self.states = {
                (pos, err)
                for pos in range(len(self.word) + 1)
                for err in range(distance + 1)
            }
            if alphabet == None:
                self.alphabet = list(set(self.word))
            else:
                self.alphabet = alphabet

            for err in range(distance, -1, -1):
                for pos in range(len(self.word), -1, -1):

                    tr = {letter:[] for letter in self.alphabet}
                    #For deletion (after converting eps-transitions to normal ones)
                    if pos < len(self.word) and err < distance:
                        tr=copy.deepcopy(self.transitions[(pos+1, err+1)])

                    for letter in self.alphabet:
                        if pos == len(self.word):
                            if err < distance:
                                tr[letter].append((pos, err+1))
                        elif letter == self.word[pos]:
                            #No change
                            if pos < len(self.word):
                                tr[letter].append((pos+1, err))

                        elif letter != self.word[pos]:
                            #For insertion 
                            if err < distance:
                                tr[letter].append((pos, err+1))

                            #For replacement
                            if pos < len(self.word) and err < distance:
                                tr[letter].append((pos+1, err+1))

                    self.transitions[(pos, err)] = copy.deepcopy(dict(tr))
            self.final_states = [(pos, err) for (pos,err) in self.states if len(self.word)-pos+err <= distance]

        self.number_of_words = {state: 0 for state in self.states}

    def __str__(self):
        return str(self.init_state) + str(self.transitions) + str(self.final_states)

    def is_word_in(self, word):
        """
        returns whether a word belongs to the levensteinNFA
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

    def generate_all_accepting_words(self):
        """
        returns all words that are accepted by the levensteinNFA
        """
        return self.generate_accepting_words(self.init_state)

    def generate_accepting_words(self, state):
        """
        returns all words that are accepted by a levensteinNFA from a given state
        """

        all_words = []
        if state in self.final_states:
            all_words += [""]

        for letter in self.alphabet:
            successor_states = self.transitions[state][letter]

            # workaround for too many words
            if len(all_words) > 100:
                break

            for next_state in successor_states:
                all_words += [
                    letter + word for word in self.generate_accepting_words(next_state)
                ]

        return all_words

    def generate_num_accepting_words(self, length):
        """
        returns the number of words that are accepted of a particular length
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
        random_length = random.randint(len(self.word)-self.distance, len(self.word)+self.distance)
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
                    if self.number_of_words[(state, length - i + 1)] == 0:
                        continue
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
        checks inclusion of the language of levensteinNFA in the language of a given DFA
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
        change_choices = ["insertion", "deletion", "replacement"]
        choice = random.choice(change_choices)
        word = og_word
        for _ in range(distance):

            if word == "":
                break
            pos = random.randint(0, len(word))
            if choice == "deletion":
                word = word[:pos] + word[pos + 1 :]

            if choice == "replacement":
                random_letter = random.choice(real_alphabet)
                word = word[:pos] + random_letter + word[pos + 1 :]

            if choice == "insertion":
                random_letter = random.choice(real_alphabet)
                word = word[:pos] + random_letter + word[pos:]

        return word

def levenstein_distance(word1, word2):
    """
    returns the levenstein distance between two words
    """

    if word1 == "":
        return len(word2)
    if word2 == "":
        return len(word1)

    if word1[0] == word2[0]:
        return levenstein_distance(word1[1:], word2[1:])
    else:
        return 1 + min(
            levenstein_distance(word1[1:], word2),
            levenstein_distance(word1, word2[1:]),
            levenstein_distance(word1[1:], word2[1:]),
        )

