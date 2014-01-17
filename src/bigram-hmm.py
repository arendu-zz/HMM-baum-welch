__author__ = 'arenduchintala'
from collections import defaultdict
import os
from pprint import pprint
from math import exp, log

START_STATE = 'START_STATE'
END_STATE = 'END_STATE'
START_WORD = '###'

global CURRENT_MLEs
CURRENT_MLEs = defaultdict(float)
global possible_states
possible_states = defaultdict(set)
possible_states[START_WORD] = set([START_STATE])


def make_mle_estimates(filepath):
    tagged_sentences = open(filepath, 'r').read().split('###/###' + os.linesep)
    for tagged_sentence in tagged_sentences:
        prev_state = 'START_STATE'
        if tagged_sentence.strip() != '':
            for state_obs in tagged_sentence.strip().split(os.linesep):
                obs = state_obs.split('/')[0]
                state = state_obs.split('/')[1]
                CURRENT_MLEs[('emission', obs, state)] += 1
                CURRENT_MLEs[('any_emission_from', state)] += 1
                possible_states[obs].add(state)
                CURRENT_MLEs[('transition', state, prev_state )] += 1
                CURRENT_MLEs[('any_transition_from', prev_state)] += 1
                prev_state = state
            CURRENT_MLEs[('transition', END_STATE, prev_state)] += 1
            CURRENT_MLEs[('any_transition_from', prev_state)] += 1  # counts the transition from a state -> END_STATE
            #convert to probabilities
    for k in CURRENT_MLEs:
        #print k, counts[k]
        if k[0] == 'emission':
            state = k[2]
            k_any = ('any_emission_from', state)
            CURRENT_MLEs[k] = CURRENT_MLEs[k] / float(CURRENT_MLEs[k_any])
        elif k[0] == 'transition':
            prev_state = k[2]
            k_any = ('any_transition_from', prev_state)
            CURRENT_MLEs[k] = CURRENT_MLEs[k] / float(CURRENT_MLEs[k_any])
        print k, CURRENT_MLEs[k]


def get_viterbi_sequence(words):
    words[0] = START_WORD
    pi = {(0, START_STATE): 1.0}
    alpha_pi = {(0, START_STATE): 1.0}
    #pi[(0, START_STATE)] = 1.0  # 0,START_STATE
    arg_pi = {(0, START_STATE): []}
    for k in range(1, max(words) + 1):  # the words are numbered from 1 to n, 0 is special start character
        for v in possible_states[words[k]]: #[1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = {}
            for u in possible_states[words[k - 1]]:#[1]:
                q = CURRENT_MLEs[('transition', v, u)] #    q(v|u)
                if ('emission', words[k][0], v) in CURRENT_MLEs:
                    e = CURRENT_MLEs[('emission', words[k][0], v)]
                else:
                    e = 0.0  # float("-inf")
                pi_key = str(k - 1) + ',' + str(u)
                p = pi[(k - 1, u)] * q * e
                alpha_p = alpha_pi[(k - 1, u)] * q * e
                #print pi_key, v, u, p, '=', q, '*', e, '*', pi[(k - 1, u)]
                bt = list(arg_pi[(k - 1, u)])  # list(list1) makes copy of list1
                bt.append(u)
                max_prob_to_bt[p] = bt
                sum_prob_to_bt[alpha_p] = None # TODO: does alpha really need back pointers?

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            print 'mu   ', new_pi_key, '=', pi[new_pi_key]
            alpha_pi[new_pi_key] = sum(sum_prob_to_bt)
            print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key]
            arg_pi[new_pi_key] = max_bt

    k = max(words)
    max_prob_to_bt = {}
    for u in possible_states[words[k]]:#[1]:
        if ('transition', END_STATE, u) in CURRENT_MLEs:  # bigramTransitions.has_key((s2i['STOP'] , v)):
            q = CURRENT_MLEs[('transition', END_STATE, u)] # bigramTransitions[(s2i['STOP'] , v)]
        else:
            q = float("-inf")
        p = pi[(k, u)] * q
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
    i = 1
    for obs in sentence:
        if obs.strip() == '':
            print 'skipping empty word'
        else:
            if obs.strip() in possible_states:  # (seenObservations.has_key(word)):
                words[i] = obs.strip()

            else:
                #TODO: unseen observation!
                pass

            i += 1
    return words


def read_test_sentences(filepath):
    test_sentences = []
    test_tags = []
    tagged_sentences = open(filepath, 'r').read().split('###/###' + os.linesep)
    for tagged_sentence in tagged_sentences:
        sentence = []
        tags = []
        if tagged_sentence.strip() != '':
            for state_obs in tagged_sentence.strip().split(os.linesep):
                sentence.append(state_obs.split('/')[0])
                tags.append(state_obs.split('/')[1])
            test_sentences.append(sentence)
            test_tags.append(tags)
    return test_sentences, test_tags


make_mle_estimates('../data/ictrain')
pprint(CURRENT_MLEs)
test_sentences, test_tags = read_test_sentences('../data/ictest')
all_predictions = [START_STATE]
all_answer_tags = [START_STATE]
all_perpexity = []
correct_tags = 0
total_tags = 0
for i in range(len(test_sentences)):
    sent = test_sentences[i]
    answer_tags = test_tags[i]
    obs = get_observations_with_indexes(sent)
    predicted_tags, max_p = get_viterbi_sequence(obs)
    print predicted_tags
    print answer_tags
    correct_indexes = [i for i in range(len(predicted_tags)) if predicted_tags[i] == answer_tags[i]]
    correct_tags += len(correct_indexes)
    total_tags += len(answer_tags)
    all_perpexity.append(exp(-log(max_p) / float(len(predicted_tags) + 1)))
    all_predictions += predicted_tags
    all_answer_tags += answer_tags


#print all_answer_tags
#print all_predictions
print 'perpexity', sum(all_perpexity) / float(len(all_perpexity))
print 'tagging accuracy:', correct_tags, '/', total_tags, '=', float(correct_tags) / float(total_tags)



