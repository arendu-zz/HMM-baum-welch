__author__ = 'arenduchintala'
from collections import defaultdict
import pdb
from pprint import pprint
from sys import argv, stderr
from math import exp, log

BOUNDRY_STATE = '###'
BOUNDRY_WORD = '###'
OOV = '**OOV**'
OBS_VOCAB = 'observation vocab'
STATE_VOCAB = 'state vocab'
N1 = 'N+1'

global CURRENT_MLEs
CURRENT_MLEs = {}
global one_count_transition
one_count_transition = {}
global one_count_emission
one_count_emission = {}
global possible_states
possible_states = defaultdict(set)
possible_states[BOUNDRY_WORD] = set([BOUNDRY_STATE])
global singleton_emissions
global singleton_transitions
singleton_transitions = defaultdict(set)
singleton_emissions = defaultdict(set)
global add_to_lambda
add_to_lambda = 1e-20


def logadd(x, y):
    """
    trick to add probabilities in logspace
    without underflow
    """
    #Todo: handle special case when x,y=0
    if x == 0.0 and y == 0.0:
        pdb.set_trace()
    if x >= y:
        return x + log(1 + exp(y - x))
    else:
        return y + log(1 + exp(x - y))


def logadd_of_list(a_list):
    logsum = a_list[0]
    for i in a_list[1:]:
        logsum = logadd(logsum, i)
    return logsum


def flatten_backpointers(bt):
    reverse_bt = []
    while len(bt) > 0:
        x = bt.pop()
        reverse_bt.append(x)
        if len(bt) > 0:
            bt = bt.pop()
    reverse_bt.reverse()
    return reverse_bt


def get_one_count_transition(v, u):
    #print 'getting one count for transition', v, u

    if ('ocp_transition', v, u) not in one_count_transition:
        one_count_lambda = len(singleton_transitions[u])
        one_count_lambda = add_to_lambda if one_count_lambda == 0 else one_count_lambda
        count_uv = CURRENT_MLEs.get(('transition', v, u), 0.0)
        count_u = CURRENT_MLEs[('count_state', u)] if u != BOUNDRY_STATE else (CURRENT_MLEs[('any_transition_from', u)] )
        count_v = CURRENT_MLEs[('count_state', v)] if v != BOUNDRY_STATE else (CURRENT_MLEs[('any_transition_from', v)] )
        p_v_unsmoothed = count_v / (CURRENT_MLEs[N1] - 1.0)
        ocp = (count_uv + one_count_lambda * p_v_unsmoothed) / float(count_u + one_count_lambda)
        if ocp == 0:
            pdb.set_trace()
            raise "One Count Smoothed Probability is for transition 0!!"
        one_count_transition[('ocp_transition', v, u)] = log(ocp)

    return one_count_transition[('ocp_transition', v, u)]


def get_one_count_emission(obs, v):
    #print 'getting one count for emission', obs, v
    if ('ocp_emission', obs, v) not in one_count_emission:
        if obs == BOUNDRY_WORD and v == BOUNDRY_STATE:
            ocp = 1.0
        elif v == BOUNDRY_STATE:
            ocp = 0.0
        else:
            one_count_lambda = len(singleton_emissions[v])
            one_count_lambda = add_to_lambda if one_count_lambda == 0 else one_count_lambda
            if obs not in CURRENT_MLEs[OBS_VOCAB]:
                V = len(CURRENT_MLEs[OBS_VOCAB])
                p_w_addone = 1.0 / float(CURRENT_MLEs[N1] - 1 + V)
            else:
                V = len(CURRENT_MLEs[OBS_VOCAB])
                p_w_addone = (CURRENT_MLEs[('count_obs', obs)] + 1.0) / float(CURRENT_MLEs[N1] - 1 + V)
            count_obs_v = CURRENT_MLEs.get(('emission', obs, v), 0.0)
            count_v = CURRENT_MLEs[('any_emission_from', v)]
            if count_v == 0:
                pdb.set_trace()
            ocp = (count_obs_v + one_count_lambda * p_w_addone) / float(count_v + one_count_lambda)
            print 'OCE:', ocp, V, p_w_addone, count_v, count_obs_v, obs, v
            #pdb.set_trace()
        if ocp == 0 and v != BOUNDRY_STATE and obs != BOUNDRY_WORD:
            pdb.set_trace()
            raise "One Count Smoothed Probability for emission is 0!!"
        one_count_emission[('ocp_emission', obs, v)] = log(ocp) if ocp > 0.0 else float('-inf')

    return one_count_emission[('ocp_emission', obs, v)]


def try_and_increment_MLE(key, val):
    try:
        CURRENT_MLEs[key] += val
    except KeyError:
        CURRENT_MLEs[key] = val


def get_possible_states(obs):
    if obs in possible_states:
        return possible_states[obs]
    else:
        return possible_states[OOV] - possible_states[BOUNDRY_WORD]


def make_mle_estimates(filepath):
    global CURRENT_MLEs
    CURRENT_MLEs[OBS_VOCAB] = set([OOV])
    CURRENT_MLEs[STATE_VOCAB] = set([])
    CURRENT_MLEs['N+1'] = 0.0
    tagged_tokens = open(filepath, 'r').readlines()
    prev_state = None
    for state_obs in tagged_tokens:
        state_obs = state_obs.strip()
        if state_obs != '':
            CURRENT_MLEs['N+1'] += 1
            obs = state_obs.split('/')[0]
            state = state_obs.split('/')[1]
            CURRENT_MLEs[OBS_VOCAB].add(obs)
            CURRENT_MLEs[STATE_VOCAB].add(state)
            try_and_increment_MLE(('count_obs', obs), 1)
            try_and_increment_MLE(('count_state', state), 1)
            try_and_increment_MLE(('any_emission_from', state), 1)
            try_and_increment_MLE(('emission', obs, state), 1)
            if CURRENT_MLEs[('emission', obs, state)] == 1:
                #this is a singleton wt
                singleton_emissions[state].add(obs)
            elif CURRENT_MLEs[('emission', obs, state)] == 2:
                #just lost a singleton
                singleton_emissions[state].remove(obs)

            possible_states[obs].add(state)
            if prev_state is not None:
                try_and_increment_MLE(('transition', state, prev_state), 1)
                try_and_increment_MLE(('any_transition_from', prev_state), 1)
                if CURRENT_MLEs[('transition', state, prev_state)] == 1:
                    #just got a singleton transition
                    singleton_transitions[prev_state].add(state)
                elif CURRENT_MLEs[('transition', state, prev_state)] == 2:
                    #just lost a singleton
                    singleton_transitions[prev_state].remove(state)
            prev_state = state
    possible_states[OOV] = CURRENT_MLEs[STATE_VOCAB]
    temp_MLE = defaultdict()
    for k in CURRENT_MLEs:  # convert to probabilities
        #print k, counts[k]
        if k[0] == 'emission':
            obs = k[1]
            state = k[2]
            k_any = ('any_emission_from', state)
            k_prob = ('emission_prob', obs, state)
            temp_MLE[k_prob] = log(CURRENT_MLEs[k] / float(CURRENT_MLEs[k_any]))
        elif k[0] == 'transition':
            state = k[1]
            prev_state = k[2]
            k_any = ('any_transition_from', prev_state)
            k_prob = ('transition_prob', state, prev_state )
            temp_MLE[k_prob] = log(CURRENT_MLEs[k] / float(CURRENT_MLEs[k_any]))
    CURRENT_MLEs = dict(CURRENT_MLEs.items() + temp_MLE.items())
    print CURRENT_MLEs[N1]
    pdb.set_trace()


def get_backwards(words, alpha_pi):
    n = len(words) - 1 # index of last word
    #words[n] = BOUNDRY_WORD  # actually the end word
    beta_pi = {(n, BOUNDRY_STATE): 0.0}
    posterior_unigrams = {}
    S = alpha_pi[(n, BOUNDRY_STATE)] # from line 13 in pseudo code
    for k in range(n - 1, -1, -1):
        for v in get_possible_states(words[k]):
            sum_prob_to_bt = []
            for u in get_possible_states(words[k + 1]):
                #print 'reverse transition', 'k', k, 'u', u, '->', 'v', v
                q = get_one_count_transition(u, v)
                e = get_one_count_emission(words[k + 1], u)
                #beta_p = beta_pi[(k + 1, u)] * q * e
                #sum_prob_to_bt.append(beta_p)
                beta_p = beta_pi[(k + 1, u)] + q + e
                sum_prob_to_bt.append(beta_p) # TODO convert this using logadd : see hints in handout
            new_pi_key = (k, v)
            #beta_pi[new_pi_key] = log(sum(sum_prob_to_bt))
            beta_pi[new_pi_key] = logadd_of_list(sum_prob_to_bt)
            posterior_unigrams[new_pi_key] = beta_pi[new_pi_key] + alpha_pi[new_pi_key] - S
            #beta_pi[new_pi_key] = sum(sum_prob_to_bt)
            print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
            print 'posterior', new_pi_key, '=', posterior_unigrams[new_pi_key], exp(posterior_unigrams[new_pi_key])
            #print 'beta', new_pi_key, '=', beta_pi[new_pi_key]
    pdb.set_trace()
    return beta_pi, posterior_unigrams


def get_viterbi_sequence(words):
    pi = {(0, BOUNDRY_STATE): 0.0}
    alpha_pi = {(0, BOUNDRY_STATE): 0.0}
    #pi[(0, START_STATE)] = 1.0  # 0,START_STATE
    arg_pi = {(0, BOUNDRY_STATE): []}
    for k in range(1, len(words)):  # the words are numbered from 1 to n, 0 is special start character
        for v in get_possible_states(words[k]): #[1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in get_possible_states(words[k - 1]):#[1]:
                q = get_one_count_transition(v, u)
                e = get_one_count_emission(words[k], v)
                #p = pi[(k - 1, u)] * q * e
                #alpha_p = alpha_pi[(k - 1, u)] * q * e
                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                if len(arg_pi[(k - 1, u)]) == 0:
                    bt = [u]
                else:
                    bt = [arg_pi[(k - 1, u)], u]
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(alpha_p) # TODO: does alpha really need back pointers?

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            #print 'mu   ', new_pi_key, '=', pi[new_pi_key]
            print 'mu   ', new_pi_key, '=', pi[new_pi_key], exp(pi[new_pi_key])
            #alpha_pi[new_pi_key] = sum(sum_prob_to_bt)
            #alpha_pi[new_pi_key] = log(sum(sum_prob_to_bt))  # sum the real probabilities, then take the log of the sum
            alpha_pi[new_pi_key] = logadd_of_list(sum_prob_to_bt)

            ##print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key]
            print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key], exp(alpha_pi[new_pi_key])
            arg_pi[new_pi_key] = max_bt

    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt = flatten_backpointers(max_bt)
    #max_bt.pop(0)
    return max_bt, max_p, alpha_pi


def get_known_indexes(test_obs):
    i = 0
    known = []
    for obs in test_obs:
        obs = obs.strip()
        if obs == BOUNDRY_WORD:
            pass
        elif obs in possible_states:  # (seenObservations.has_key(word)):
            known.append(1)
        else:
            known.append(0)
        i += 1
    return known


def read_test_sentences(filepath):
    test_tags = []
    test_obs = []
    tagged_tokens = open(filepath, 'r').readlines()
    for state_obs in tagged_tokens:
        state_obs = state_obs.strip()
        if state_obs != '':
            test_tags.append(state_obs.split('/')[1])
            test_obs.append(state_obs.split('/')[0])
    return test_tags, test_obs


def posterior_decoding(alphas, betas):
    n = len(alphas)
    S = alphas[(n, BOUNDRY_STATE)]


if __name__ == "__main__":
    try:
        train_file = argv[1]
        test_file = argv[2]
    except:
        train_file = '../data/ic2train'
        test_file = '../data/ictest2'

    make_mle_estimates(train_file)
    answer_tags, test_obs = read_test_sentences(test_file)
    answer_tags = filter(lambda x: x != BOUNDRY_STATE, answer_tags)
    correct_tags = 0
    total_tags = 0
    correct_known_tags = 0.0
    total_known_tags = 0.0
    known = get_known_indexes(test_obs)
    predicted_tags, max_p, alpha_pi = get_viterbi_sequence(test_obs)
    beta_pi, posterior_probabilities = get_backwards(test_obs, alpha_pi)
    num_sentences = len(predicted_tags)
    predicted_tags = filter(lambda x: x != BOUNDRY_STATE, predicted_tags)
    num_sentences -= len(predicted_tags) # number of sentences is the number of boundry words detected and removed
    correct_indexes = [i for i in range(len(predicted_tags)) if predicted_tags[i] == answer_tags[i]]
    correct_tags += len(correct_indexes)
    total_tags += len(answer_tags)
    correct_known_indexes = [idx for idx, i in enumerate(known) if (i == 1 and answer_tags[idx] == predicted_tags[idx])]

    correct_known_tags += len(correct_known_indexes)
    total_known_tags += sum(known)

    known_accuracy = 100 * correct_known_tags / float(total_known_tags)
    try:
        unknown_accuracy = 100 * (correct_tags - correct_known_tags) / float(total_tags - total_known_tags)
    except ZeroDivisionError:
        unknown_accuracy = 0.0

    #print len(predicted_tags), CURRENT_MLEs[N1]
    all_perpexity = (exp(-max_p / float(len(predicted_tags) + num_sentences)))
    print 'calc:', max_p, len(predicted_tags), num_sentences
    tagging_accuracy = "%.2f" % (100 * float(correct_tags) / float(total_tags))
    perplexity_per_word = "%.10f" % (all_perpexity)
    stderr.write(str('Tagging accuracy (Viterbi decoding): ' + str(tagging_accuracy) + '%\t'))
    stderr.write(str('(known: ' + str("%.2f" % known_accuracy) + '%\t'))
    stderr.write(str('novel: ' + str("%.2f" % unknown_accuracy) + '%)\n'))
    stderr.write('Perplexity per Viterbi-tagged test word: ' + perplexity_per_word)
    stderr.write('\n')