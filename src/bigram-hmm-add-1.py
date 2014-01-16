__author__ = 'arenduchintala'
from collections import defaultdict
from os import linesep
from math import log, exp
from sys import stderr, argv
import pdb

START_STATE = 'START_STATE'
END_STATE = 'END_STATE'
START_WORD = '###'

global add_lambda
add_lambda = 0
global CURRENT_MLEs
CURRENT_MLEs = defaultdict(float)
global possible_states
possible_states = defaultdict(set)
possible_states[START_WORD] = set([START_STATE])


def make_mle_estimates(filepath):
    CURRENT_MLEs[('observations',)] = set([])  # these are also tuples
    CURRENT_MLEs[('states',)] = set([])
    tagged_sentences = open(filepath, 'r').read().split('###/###' + linesep)
    for tagged_sentence in tagged_sentences:
        prev_state = 'START_STATE'
        if tagged_sentence.strip() != '':
            for state_obs in tagged_sentence.strip().split(linesep):
                obs = state_obs.split('/')[0]
                state = state_obs.split('/')[1]
                CURRENT_MLEs[('observations',)].update([obs])  # has a set of all seen observations
                CURRENT_MLEs[('states',)].update([state])  # has a set of all seen states
                CURRENT_MLEs[('emission', obs, state)] += 1
                CURRENT_MLEs[('any_emission_from', state)] += 1
                possible_states[obs].add(state)
                CURRENT_MLEs[('transition', state, prev_state)] += 1
                CURRENT_MLEs[('any_transition_from', prev_state)] += 1
                prev_state = state
            CURRENT_MLEs[('transition', END_STATE, prev_state)] += 1
            CURRENT_MLEs[('any_transition_from', prev_state)] += 1  # counts the transition from a state -> END_STATE
            CURRENT_MLEs[(('observations',))].update(['O_O_V'])
    for k in CURRENT_MLEs:  # convert to log probabilities
        #print k, counts[k]
        if k[0] == 'emission':
            state = k[2]
            k_any = ('any_emission_from', state)
            CURRENT_MLEs[k] = log((CURRENT_MLEs[k] + add_lambda) / float(CURRENT_MLEs[k_any] + add_lambda * len(CURRENT_MLEs[('observations',)])))
        elif k[0] == 'transition':
            prev_state = k[2]
            k_any = ('any_transition_from', prev_state)
            CURRENT_MLEs[k] = log((CURRENT_MLEs[k] + add_lambda) / float(CURRENT_MLEs[k_any] + add_lambda * len(CURRENT_MLEs[('states',)])))

    for a_state in CURRENT_MLEs[('states',)]:  # give appropriate add-lambda smoothed probabilities to O_O_V
        k_any = ('any_emission_from', a_state)
        if add_lambda > 0:
            CURRENT_MLEs[('emission', 'O_O_V', a_state)] = log(
                add_lambda / float(CURRENT_MLEs[k_any] + add_lambda * len(CURRENT_MLEs[('observations',)])))
        else:
            CURRENT_MLEs[('emission', 'O_O_V', a_state)] = float("-inf")


def get_viterbi_sequence(words):
    words[0] = START_WORD
    pi = {(0, START_STATE): 0.0}
    alpha_pi = {(0, START_STATE): 0.0}
    arg_pi = {(0, START_STATE): []}
    for k in range(1, max(words) + 1):  # the words are numbered from 1 to n, 0 is special start character
        for v in possible_states[words[k]]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in possible_states[words[k - 1]]:  

                if ('transition', v, u) in CURRENT_MLEs:
                    q = CURRENT_MLEs[('transition', v, u)]  # q(v|u)
                else:
                    print 'unseen transition', u, ' to ', v
                    if add_lambda == 0:
                        q = float("-inf")
                    else:
                        q = add_lambda / float(CURRENT_MLEs[('any_transition_from', u)] + add_lambda * len(CURRENT_MLEs[('states',)]))

                if words[k] in CURRENT_MLEs[('observations',)]:
                    if ('emission', words[k], v) in CURRENT_MLEs:
                        e = CURRENT_MLEs[('emission', words[k], v)]
                    else:
                        e = float("-inf")
                else:  # novel word
                    e = CURRENT_MLEs[('emission', 'O_O_V', v)]
                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                #print pi_key, v, u, p, '=', q, '*', e, '*', pi[(k - 1, u)]
                bt = list(arg_pi[(k - 1, u)])  # list(list1) makes copy of list1
                bt.append(u)
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(exp(alpha_p))

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            print 'mu   ', new_pi_key, '=', exp(pi[new_pi_key])
            alpha_pi[new_pi_key] = log(sum(sum_prob_to_bt))  # sum the real probabilities, then take the log of the sum
            print 'alpha', new_pi_key, '=', exp(alpha_pi[new_pi_key])
            arg_pi[new_pi_key] = max_bt

    k = max(words)
    max_prob_to_bt = {}
    for u in possible_states[words[k]]:
        if ('transition', END_STATE, u) in CURRENT_MLEs:
            q = CURRENT_MLEs[('transition', END_STATE, u)]
        else:
            q = float("-inf")
        p = pi[(k, u)] + q
        bt = list(arg_pi[(k, u)])
        bt.append(u)
        max_prob_to_bt[p] = bt
    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt.pop(0)
    return max_bt, max_p


def get_observations_with_indexes(sentence):
    """
    Simply returns a dict with observations as values, and indexes to observations as keys
    Adds 0 index as the START_STATE
    e.g. observations : '1','2','3','2','2'
    will be returned as: { 1:'1', 2:'2', 3:'2', 4:'2', 5:'2'}
    """
    words = {}
    known = []
    i = 0
    for obs in sentence:
        if obs.strip() == '':
            print 'skipping empty word'
        else:
            i += 1
            if obs.strip() in possible_states:  # (seenObservations.has_key(word)):
                words[i] = obs.strip()
                known.append(1)
            else:
                possible_states[obs.strip()] = CURRENT_MLEs[('states',)] # giving unseen observation all possible states
                words[i] = obs.strip()
                known.append(0)

    return words, known


def read_test_sentences(filepath):
    test_sentences = []
    test_tags = []
    tagged_sentences = open(filepath, 'r').read().split('###/###' + linesep)
    for tagged_sentence in tagged_sentences:
        sentence = []
        tags = []
        if tagged_sentence.strip() != '':
            for state_obs in tagged_sentence.strip().split(linesep):
                sentence.append(state_obs.split('/')[0])
                tags.append(state_obs.split('/')[1])
            test_sentences.append(sentence)
            test_tags.append(tags)
    return test_sentences, test_tags


if __name__ == "__main__":
    try:
        train_file = argv[1]
        test_file = argv[2]
    except:
        train_file = '../data/ictrain'
        test_file = '../data/ictest2'

    try:
        add_lambda = float(argv[3])
    except:
        add_lambda = 0
    make_mle_estimates(train_file)
    test_sentences, test_tags = read_test_sentences(test_file)
    all_predictions = [START_STATE]
    all_answer_tags = [START_STATE]
    correct_tags = 0
    correct_known_tags = 0
    total_known_tags = 0
    total_tags = 0
    for i in range(len(test_sentences)):
        sent = test_sentences[i]
        print sent
        answer_tags = test_tags[i]
        obs, known = get_observations_with_indexes(sent)
        predicted_tags, best_mu = get_viterbi_sequence(obs)
        print predicted_tags
        print answer_tags
        print exp(best_mu), '\n'
        correct_indexes = [i for i in range(len(predicted_tags)) if predicted_tags[i] == answer_tags[i]]
        correct_known_indexes = [idx for idx, i in enumerate(known) if
                                 (i == 1 and answer_tags[idx] == predicted_tags[idx])]
        correct_tags += len(correct_indexes)
        total_tags += len(answer_tags)
        correct_known_tags += len(correct_known_indexes)
        total_known_tags += sum(known)

    stderr.write('Perplexity per Viterbi-tagged test word: ' + str(exp(-best_mu / float(len(answer_tags) + 1))))
    stderr.write('\n')
    print 'tagging accuracy:', correct_tags, '/', total_tags, '=', float(correct_tags) / float(total_tags)
    print 'tagging accuracy known:', correct_known_tags, '/', total_known_tags, '=', correct_known_tags / float(
        total_known_tags)
    stderr.write(str('Tagging accuracy (Viterbi decoding): ' + str(float(correct_tags) / float(total_tags))))
    stderr.write('\n')



