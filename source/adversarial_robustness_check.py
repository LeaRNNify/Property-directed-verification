import os
import time
import csv
import random
import numpy as np

from dfa import (
    DFA,
    complement,
    intersection_dfa_nfa,
    generate_all_accepting_words,
)
from levenstein_automaton import LevensteinNFA
from hamming_automaton import HammingNFA
from random_words import random_nonempty_word
from model_checker import *
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import RNNLanguageClasifier
from pac_teacher import PACTeacher

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

confidence = 0.01
error = 0.01

# Information for noting the results in csv format
Adv_check_info = [
    "Example name",
    "Word Length",
    "Distance",
    "Method",
    "Adv Example Length",
    "Time taken",
    "DFA Size",
    "Num of examples",
    "Adv Example Counting Time",
]
Adv_check_dict = {i: None for i in Adv_check_info}
csvfile_pdv = "adv_info_pdv.csv"
csvfile_stat_nagsmc = "adv_info_stat_nagsmc.csv"
csvfile_stat_smc = "adv_info_stat_smc.csv"
Adv_check_info_extra = [
    "Example name",
    "Word Length",
    "Distance",
    "Method",
    "Time taken",
    "Num of examples",
]
count_adv = False


def write_csv_header(csvfile, header):

    with open(csvfile, mode="w") as file:
        writer = csv.DictWriter(file, fieldnames=Adv_check_info)
        writer.writeheader()


def write_line_csv(csvfile, fieldnames):

    with open(csvfile, mode="a") as file:
        writer = csv.DictWriter(file, fieldnames=Adv_check_info)
        writer.writerow(Adv_check_dict)


# finding_time: time to search for a adversarial example
# counting_time: time to count all the adversarial examples (beta version)


##############################################################
################### Property Directed ########################
##############################################################


def check_adv_robustness_and_learn(rnn, neighbourhoodNFA, TO=600):

    teacher_pac = PACTeacher(rnn)
    student = DecisionTreeLearner(teacher_pac)
    batch_size = 200

    last_noted_time = time.time()
    is_positive_word = rnn.is_word_in(neighbourhoodNFA.word)
    counting_time = time.time()

    adversarial_example = teacher_pac.adv_robustness(
        student, neighbourhoodNFA, is_positive_word, TO
    )  # calls function in pac_teacher
    finding_time = time.time() - last_noted_time
    adv_examples_count = 0

    Adv_check_dict.update(
        {
            "Method": "PDV",
            "Adv Example Length": None,
            "Time taken": finding_time,
            "DFA Size": len(student.dfa.states),
        }
    )

    if adversarial_example is None:
        print("RNN is adversarially robust")
    else:
        Adv_check_dict.update({"Adv Example Length": len(adversarial_example)})
        print(
            "!!!!!Adversarial examples found in inclusion checking!!!!!!!!",
            "".join(adversarial_example),
        )

    ade_counting_time = time.time()
    # write_line_csv()
    ########## For counting adversarial examples############
    if adversarial_example and count_adv:
        all_adversarial_examples = []
        if is_positive_word:
            prod_aut = intersection_dfa_nfa(complement(student.dfa), neighbourhoodNFA)
            accepting_words = generate_all_accepting_words(prod_aut)

            # divide accepting words in batches
            accepting_words = list(set(accepting_words))
            batches = []
            batch = []
            for word in accepting_words:
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
                batch.append(word)
            batches.append(batch)  # last batch

            for batch in batches:
                rnn_results = rnn.is_words_in_batch(batch).tolist()
                for i in range(len(rnn_results)):
                    if not rnn_results[i]:
                        all_adversarial_examples.append(batch[i])
        else:
            prod_aut = intersection_dfa_nfa(student.dfa, neighbourhoodNFA)
            accepting_words = generate_all_accepting_words(prod_aut)
            for word in accepting_words:
                if not neighbourhoodNFA.is_word_in(word):
                    print("Something is wrong")

            # divide accepting words in batches
            accepting_words = list(set(accepting_words))
            batches = []
            batch = []
            for word in accepting_words:
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
                batch.append(word)
            batches.append(batch)  # last batch

            for batch in batches:
                rnn_results = rnn.is_words_in_batch(batch).tolist()
                for i in range(len(rnn_results)):
                    if rnn_results[i]:
                        all_adversarial_examples.append(batch[i])

        adv_examples_count = len(set(all_adversarial_examples))
        Adv_check_dict.update({"Num of examples": adv_examples_count})
        Adv_check_dict.update(
            {"Adv Example Counting Time": time.time() - ade_counting_time}
        )
        print("Number of adversarial_examples:", adv_examples_count)
        print("which are: " + str(set(all_adversarial_examples)))

    counting_time = time.time() - counting_time
    print("Time elapsed: %.3f" % counting_time)
    write_line_csv(csvfile_pdv, Adv_check_info_extra)


################################################################
################## Statistical Method NAG-SMC ########################
################################################################


def stat_adv_robustness_check_nagsmc(rnn, neighbourhoodNFA, TO=600):

    test_size = np.log(2 / confidence) / (
        2 * error * error
    )  # change the confidence and width according to chernoff_hoeding

    batch_size = 100
    adversarial_example = None
    counting_time = time.time()

    is_positive_word = rnn.is_word_in(neighbourhoodNFA.word)
    og_word_length, og_word_distance = (
        len(neighbourhoodNFA.word),
        neighbourhoodNFA.distance,
    )
    neighbourhoodNFA.generate_num_accepting_words(og_word_length + og_word_distance)

    last_noted_time = time.time()
    for _ in range(int(test_size // batch_size) + 1):
        batch = []
        for _ in range(batch_size):
            batch.append(neighbourhoodNFA.generate_random_word())

        rnn_result = rnn.is_words_in_batch(batch)
        for res, w in zip(rnn_result, batch):
            if is_positive_word != res:
                adversarial_example = w
                print("!!!!!Adversarial examples found!!!!!!!!", w)
                break
        else:
            continue
        break

    finding_time = time.time() - last_noted_time
    Adv_check_dict.update(
        {
            "Method": "Stat_NAGSMC",
            "Adv Example Length": None,
            "Time taken": finding_time,
            "DFA Size": None,
        }
    )
    if adversarial_example != None:
        Adv_check_dict.update({"Adv Example Length": len(adversarial_example)})
    else:
        print("RNN adversarially robust")

    adv_counting_time = time.time()
    ##########Counting adversarial examples############
    adv_examples_count = 0
    if count_adv:
        for i in range(int(test_size // batch_size) + 1):
            batch = []
            for _ in range(batch_size):

                batch.append(neighbourhoodNFA.generate_random_word())

            rnn_result = rnn.is_words_in_batch(batch)
            for res, w in zip(rnn_result, batch):
                if is_positive_word != res:
                    adv_examples_count += 1
        Adv_check_dict.update({"Num of examples": adv_examples_count})
        Adv_check_dict.update(
            {"Adv Example Counting Time": time.time() - adv_counting_time}
        )

    counting_time = time.time() - counting_time
    print("Number of adversarial_examples:", adv_examples_count)
    print("Time elapsed: %.3f" % counting_time)
    write_line_csv(csvfile_stat_nagsmc, Adv_check_info_extra)


################################################################
################## Statistical Method SMC ########################
################################################################


def stat_adv_robustness_check_smc(rnn, neighbourhoodNFA, TO=600):

    og_word = "".join(
        neighbourhoodNFA.word
    )  # converting to string for better string operations
    og_distance = neighbourhoodNFA.distance
    og_alphabet = neighbourhoodNFA.alphabet
    test_size = np.log(2 / confidence) / (2 * error * error)
    counting_time = time.time()
    batch_size = 100
    is_positive_word = rnn.is_word_in(neighbourhoodNFA.word)
    adversarial_example = None

    last_noted_time = time.time()
    for i in range(int(test_size // batch_size) + 1):

        batch = []
        for _ in range(batch_size):

            word = neighbourhoodNFA.distance_random_word(
                og_word, og_distance, og_alphabet
            )

            batch.append(word)

        rnn_result = rnn.is_words_in_batch(batch)

        for res, w in zip(rnn_result, batch):
            if is_positive_word != res:
                adversarial_example = w
                print("!!!!!Adversarial examples found!!!!!!!!", w)
                break
        else:
            continue

        break

    finding_time = time.time() - last_noted_time
    Adv_check_dict.update(
        {
            "Method": "Stat_SMC",
            "Adv Example Length": None,
            "Time taken": finding_time,
            "DFA Size": None,
        }
    )
    if adversarial_example != None:
        Adv_check_dict.update({"Adv Example Length": len(adversarial_example)})
    else:
        print("RNN adversarially robust")

    ##########Counting adversarial examples############
    ade_counting_time = time.time()

    if count_adv:
        adv_examples_count = 0
        for i in range(int(test_size // batch_size) + 1):

            batch = []
            for _ in range(batch_size):

                word = neighbourhoodNFA.distance_random_word(
                    og_word, og_distance, og_alphabet
                )
                if not (neighbourhoodNFA.is_word_in(word)):
                    print("Something is wrong")

                batch.append(word)

            rnn_result = rnn.is_words_in_batch(batch)

            for res, w in zip(rnn_result, batch):
                if is_positive_word != res:
                    adv_examples_count += 1
        print("Number of adversarial_examples:", adv_examples_count)
        Adv_check_dict.update({"Num of examples": adv_examples_count})
        Adv_check_dict.update(
            {"Adv Example Counting Time": time.time() - ade_counting_time}
        )

    counting_time = time.time() - counting_time
    print("Time elapsed: %.3f" % counting_time)
    write_line_csv(csvfile_stat_smc, Adv_check_info_extra)


def main():

    write_csv_header(csvfile_pdv, Adv_check_info_extra)
    # write_csv_header(csvfile_stat_nagsmc, Adv_check_info_extra)
    # write_csv_header(csvfile_stat_smc, Adv_check_info_extra)

    dirs = "../models/modulo_2_bench/1"

    # Trained RNN
    rnn = RNNLanguageClasifier()
    rnn.load_lstm(dirs)

    # Training set
    training_list = []
    traning_labels = []
    with open(dirs + "/training_set.csv", "r") as f:
        training_list = f.readlines()
    with open(dirs + "/training_labels.csv", "r") as f:
        traning_labels = f.readlines()

    int_to_char = {i + 1: rnn.alphabet[i] for i in range(len(rnn.alphabet))}
    true_count = 0
    false_count = 0
    for str_word, str_label in zip(training_list, traning_labels):
        str_word = str_word.replace(",", "").strip("\n")
        if str_word == "0" or len(str_word) < 6:
            continue
        real_word = ""
        for i in range(len(str_word)):
            real_word += int_to_char[int(str_word[i])]
        str_label = str_label.strip("\n")
        real_label = str_label == "True"

        # 100 positive and 100 negative
        print(real_label)
        if real_label:
            true_count += 1
            if true_count > 100:
                continue
        else:
            false_count += 1
            if false_count > 100:
                continue
        if true_count > 100 and false_count > 100:
            return

        for distance in range(1, 6):

            neighbourhoodNFA = HammingNFA(real_word, distance, rnn.alphabet)

            Adv_check_dict.update(
                {
                    "Example name": dirs,
                    "Word Length": len(real_word),
                    "Distance": distance,
                }
            )
            print(
                "Checking for adversarial examples in distance of %d from word of length %s"
                % (distance, real_word)
            )

            print("***********************************************")
            print("Starting PDV for RNN of DFA in %s" % dirs)
            print("***********************************************")

            check_adv_robustness_and_learn(rnn, neighbourhoodNFA)

            # print("***********************************************")
            # print("Starting Stat2 for RNN of DFA in %s" % dirs)
            # print("***********************************************")

            # stat_adv_robustness_check_nagsmc(rnn, neighbourhoodNFA)

            # print("***********************************************")
            # print("Starting Stat3 for RNN of DFA in %s" % dirs)
            # print("***********************************************")

            # stat_adv_robustness_check_smc(rnn, neighbourhoodNFA)



if __name__ == "__main__":
    main()
