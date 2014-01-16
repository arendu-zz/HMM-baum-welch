__author__ = 'arenduchintala'
from collections import defaultdict
from copy import deepcopy
from sys import argv, stderr
from math import exp, log

BOUNDRY_STATE = '###'
BOUNDRY_WORD = '###'
OOV = '**OOV**'
TRAINING_OBS_VOCAB = 'observation vocab'
RAW_OBS_VOCAB = 'raw observation vocab'
ALL_OBS_VOCAB = 'all observation vocab'
STATE_VOCAB = 'state vocab'
N1 = 'N+1'
STATUS_ORIGINAL = 'original'
STATUS_CURRENT = 'current'
STATUS_NEW = 'new'

global CURRENT_MLEs
CURRENT_MLEs = {}
global ORIGINAL_MLEs
ORIGINAL_MLEs = {}
global NEW_MLEs
NEW_MLEs = {}
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
        return log(exp(x) + exp(y))
    elif x >= y:
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
    ##print 'getting one count for transition', v, u
    if ('ocp_transition', v, u) not in one_count_transition:
        one_count_lambda = len(singleton_transitions[u])
        one_count_lambda = add_to_lambda if one_count_lambda == 0 else one_count_lambda
        count_uv = CURRENT_MLEs.get(('count_transition', v, u), 0.0)
        count_u = CURRENT_MLEs[('count_state', u)] if u != BOUNDRY_STATE else (CURRENT_MLEs[('any_transition_from', u)] )
        count_v = CURRENT_MLEs[('count_state', v)] if v != BOUNDRY_STATE else (CURRENT_MLEs[('any_transition_from', v)] )
        p_v_unsmoothed = count_v / (CURRENT_MLEs[N1] - 1.0)
        if u == BOUNDRY_STATE:
            pass
        ocp = (count_uv + one_count_lambda * p_v_unsmoothed) / float(count_u + one_count_lambda)
        if ocp == 0:
            raise BaseException("One Count Smoothed Probability is for transition 0!!")
        one_count_transition[('ocp_transition', v, u)] = log(ocp)
    return one_count_transition[('ocp_transition', v, u)]


def get_one_count_emission(obs, v):
    ##print 'getting one count for emission', obs, v
    if ('ocp_emission', obs, v) not in one_count_emission:
        if obs == BOUNDRY_WORD and v == BOUNDRY_STATE:
            ocp = 1.0
        elif v == BOUNDRY_STATE:
            ocp = 0.0
        else:
            one_count_lambda = len(singleton_emissions[v])
            one_count_lambda = add_to_lambda if one_count_lambda == 0 else one_count_lambda
            if obs not in CURRENT_MLEs[TRAINING_OBS_VOCAB]:
                V = len(CURRENT_MLEs[ALL_OBS_VOCAB])
                p_w_addone = 1.0 / float(CURRENT_MLEs[N1] - 1 + V)
            else:
                V = len(CURRENT_MLEs[ALL_OBS_VOCAB])
                p_w_addone = (CURRENT_MLEs[('count_obs', obs)] + 1.0) / float(CURRENT_MLEs[N1] - 1 + V)
            count_obs_v = CURRENT_MLEs.get(('count_emission', obs, v), 0.0)
            count_v = CURRENT_MLEs[('any_emission_from', v)]
            if count_v == 0:
                raise BaseException("One count smoothed probability emission count_v is 0!!")
            ocp = (count_obs_v + one_count_lambda * p_w_addone) / float(count_v + one_count_lambda)
        if ocp == 0 and v != BOUNDRY_STATE and obs != BOUNDRY_WORD:
            raise BaseException("One Count Smoothed Probability for emission is 0!!")
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


def save_current_to_original():
    global ORIGINAL_MLEs
    global CURRENT_MLEs
    ORIGINAL_MLEs = deepcopy(CURRENT_MLEs)


def set_current_from_new():
    global NEW_MLEs
    global CURRENT_MLEs
    CURRENT_MLEs = deepcopy(NEW_MLEs)


def make_new_mle_estimates(posterior_bigrams_accumilation, posterior_obs_accumilation, include_original=False):
    #print 'updating MLEs...'
    global NEW_MLEs
    NEW_MLEs = deepcopy(CURRENT_MLEs)
    for k in posterior_obs_accumilation:
        #if k[0] == 'count_obs':
        #    #print 'added new vocab found in raw', k[1]
        #    NEW_MLEs[TRAINING_OBS_VOCAB].add(k[1])

        NEW_MLEs[k] = exp(posterior_obs_accumilation[k])
        if k in ORIGINAL_MLEs and include_original:
            NEW_MLEs[k] += ORIGINAL_MLEs[k]
        '''
        try:
            print k, CURRENT_MLEs[k], '-->', "%.3f" % (NEW_MLEs[k])
        except KeyError:
            print k, 'absent', '-->', "%.3f" % (NEW_MLEs[k])
        '''
    for k in posterior_bigrams_accumilation:
        NEW_MLEs[k] = exp(posterior_bigrams_accumilation[k])
        if include_original:
            NEW_MLEs[k] += ORIGINAL_MLEs[k]

    #print "NEW_MLE Probabilities"
    temp_MLE = convert_counts_to_log_probs(NEW_MLEs)
    NEW_MLEs = dict(NEW_MLEs.items() + temp_MLE.items())
    #print 'clearing one count values...'
    global one_count_transition
    one_count_transition = {}
    global one_count_emission
    one_count_emission = {}
    #print 'update MLEs completed...'


def add_raw_obs_to_initial_mle_estimates(raw_obs):
    global CURRENT_MLEs
    CURRENT_MLEs[RAW_OBS_VOCAB] = set(raw_obs)
    CURRENT_MLEs[ALL_OBS_VOCAB] = CURRENT_MLEs[TRAINING_OBS_VOCAB].union(CURRENT_MLEs[RAW_OBS_VOCAB])


def make_initial_mle_estimates(states, observations):
    global CURRENT_MLEs
    CURRENT_MLEs[TRAINING_OBS_VOCAB] = set([OOV])
    CURRENT_MLEs[STATE_VOCAB] = set([])
    CURRENT_MLEs['N+1'] = 0.0
    prev_state = None
    for idx, obs in enumerate(observations):
        state = states[idx].strip()
        obs = obs.strip()
        if obs != '':
            CURRENT_MLEs['N+1'] += 1
            CURRENT_MLEs[TRAINING_OBS_VOCAB].add(obs)
            CURRENT_MLEs[STATE_VOCAB].add(state)
            try_and_increment_MLE(('count_obs', obs), 1)
            try_and_increment_MLE(('count_state', state), 1)
            try_and_increment_MLE(('any_emission_from', state), 1)
            try_and_increment_MLE(('count_emission', obs, state), 1)
            if CURRENT_MLEs[('count_emission', obs, state)] == 1:
                #this is a singleton wt
                singleton_emissions[state].add(obs)
            elif CURRENT_MLEs[('count_emission', obs, state)] == 2:
                #just lost a singleton
                singleton_emissions[state].remove(obs)

            possible_states[obs].add(state)
            if prev_state is not None:
                try_and_increment_MLE(('count_transition', state, prev_state), 1)
                try_and_increment_MLE(('any_transition_from', prev_state), 1)
                if CURRENT_MLEs[('count_transition', state, prev_state)] == 1:
                    #just got a singleton transition
                    singleton_transitions[prev_state].add(state)
                elif CURRENT_MLEs[('count_transition', state, prev_state)] == 2:
                    #just lost a singleton
                    singleton_transitions[prev_state].remove(state)
            prev_state = state
    possible_states[OOV] = CURRENT_MLEs[STATE_VOCAB]
    temp_MLE = convert_counts_to_log_probs(CURRENT_MLEs)
    CURRENT_MLEs = dict(CURRENT_MLEs.items() + temp_MLE.items())


def convert_counts_to_log_probs(MLEs):
    temp_MLE = defaultdict()
    for k in MLEs:  # convert to probabilities
        #print k, counts[k]
        if k[0] == 'count_emission':
            obs = k[1]
            state = k[2]
            k_any = ('any_emission_from', state)
            k_prob = ('emission_prob', obs, state)
            temp_MLE[k_prob] = log(MLEs[k] / float(MLEs[k_any]))
            #print k_prob, '-->', "%.4f" % (exp(temp_MLE[k_prob]))
        elif k[0] == 'count_transition':
            state = k[1]
            prev_state = k[2]
            k_any = ('any_transition_from', prev_state)  # TODO isn't any_transition_from == any_emission_from??
            k_prob = ('transition_prob', state, prev_state)
            temp_MLE[k_prob] = log(MLEs[k] / float(MLEs[k_any]))
            #print k_prob, '-->', "%.4f" % (exp(temp_MLE[k_prob]))
    return temp_MLE


def do_accumilate_posterior_obs(accumilation_dict, obs, state, posterior_unigram_val):
    # these are actual counts in log space!!
    if ('count_obs', obs) in accumilation_dict:
        accumilation_dict[('count_obs', obs)] = logadd(accumilation_dict[('count_obs', obs)], posterior_unigram_val)
    else:
        accumilation_dict[('count_obs', obs)] = posterior_unigram_val
    if ('count_state', state) in accumilation_dict:
        accumilation_dict[('count_state', state)] = logadd(accumilation_dict[('count_state', state)], posterior_unigram_val)
    else:
        accumilation_dict[('count_state', state)] = posterior_unigram_val

    if ('count_emission', obs, state) in accumilation_dict:
        accumilation_dict[('count_emission', obs, state)] = logadd(accumilation_dict[('count_emission', obs, state)], posterior_unigram_val)
    else:
        accumilation_dict[('count_emission', obs, state)] = posterior_unigram_val
        # doing total counts ...
    if ('any_emission_from', state) in accumilation_dict:
        accumilation_dict[('any_emission_from', state)] = logadd(accumilation_dict[('any_emission_from', state)], posterior_unigram_val)
    else:
        accumilation_dict[('any_emission_from', state)] = posterior_unigram_val
    return accumilation_dict


def do_accumilate_posterior_bigrams(accumilation_dict, v, u, posterior_bigram_val):
    # these are actual counts in log space!!
    if ('count_transition', v, u) not in accumilation_dict:
        accumilation_dict[('count_transition', v, u)] = posterior_bigram_val
    else:
        accumilation_dict[('count_transition', v, u)] = logadd(accumilation_dict[('count_transition', v, u)], posterior_bigram_val)

    if ('any_transition_from', u) not in accumilation_dict:
        accumilation_dict[('any_transition_from', u)] = posterior_bigram_val
    else:
        accumilation_dict[('any_transition_from', u)] = logadd(accumilation_dict[('any_transition_from', u)], posterior_bigram_val)
    return accumilation_dict


def do_append_posterior_unigrams(appending_dict, position, state, posterior_unigram_val):
    if position in appending_dict:
        appending_dict[position].append((state, posterior_unigram_val))
    else:
        appending_dict[position] = [(state, posterior_unigram_val)]
    return appending_dict


def get_backwards(words, alpha_pi):
    n = len(words) - 1 # index of last word
    beta_pi = {(n, BOUNDRY_STATE): 0.0}
    posterior_unigrams = {}
    posterior_obs_accumilation = {}
    posterior_bigrams_accumilation = {}
    S = alpha_pi[(n, BOUNDRY_STATE)] # from line 13 in pseudo code
    for k in range(n, 0, -1):
        for v in get_possible_states(words[k]):
            e = get_one_count_emission(words[k], v)
            pb = beta_pi[(k, v)]
            posterior_unigram_val = beta_pi[(k, v)] + alpha_pi[(k, v)] - S
            posterior_obs_accumilation = do_accumilate_posterior_obs(posterior_obs_accumilation, words[k], v, posterior_unigram_val)
            posterior_unigrams = do_append_posterior_unigrams(posterior_unigrams, k, v, posterior_unigram_val)

            for u in get_possible_states(words[k - 1]):
                #print 'reverse transition', 'k', k, 'u', u, '->', 'v', v
                q = get_one_count_transition(v, u)
                p = q + e
                beta_p = pb + p
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = logadd(beta_pi[new_pi_key], beta_p)
                    #print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                posterior_bigram_val = alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
                #posterior_bigram_val = "%.3f" % (exp(alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S))
                posterior_bigrams_accumilation = do_accumilate_posterior_bigrams(posterior_bigrams_accumilation, v, u, posterior_bigram_val)
                '''
                if k not in posterior_bigrams:
                    posterior_bigrams[k] = [((u, v), posterior_bigram_val)]
                else:
                    posterior_bigrams[k].append(((u, v), posterior_bigram_val))
                '''

    return posterior_unigrams, posterior_bigrams_accumilation, posterior_obs_accumilation, S


def get_viterbi_and_forward(words):
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
                sum_prob_to_bt.append(alpha_p)

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            #print 'mu   ', new_pi_key, '=', pi[new_pi_key], exp(pi[new_pi_key])
            alpha_pi[new_pi_key] = logadd_of_list(sum_prob_to_bt)
            #print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key], exp(alpha_pi[new_pi_key])
            arg_pi[new_pi_key] = max_bt

    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt = flatten_backpointers(max_bt)
    return max_bt, max_p, alpha_pi


def get_known_indexes(test_obs):
    known = []
    full = []
    seen = []
    novel = []
    for obs in test_obs:
        obs = obs.strip()
        if obs == BOUNDRY_WORD:
            pass
        else:
            full.append(1)
            if obs in CURRENT_MLEs[TRAINING_OBS_VOCAB]:  # (seenObservations.has_key(word)):
                known.append(1)
            else:
                known.append(0)
            if obs in CURRENT_MLEs[RAW_OBS_VOCAB] and obs not in CURRENT_MLEs[TRAINING_OBS_VOCAB]:
                seen.append(1)
            else:
                seen.append(0)
            if obs not in CURRENT_MLEs[TRAINING_OBS_VOCAB] and obs not in CURRENT_MLEs[RAW_OBS_VOCAB]:
                novel.append(1)
            else:
                novel.append(0)
    return full, known, seen, novel


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


def read_raw_sentences(filepath):
    raw_obs = []
    tokens = open(filepath, 'r').readlines()
    for obs in tokens:
        obs = obs.strip()
        if obs != '':
            raw_obs.append(obs)
    return raw_obs


def posterior_unigram_decoding(posterior_unigrams):
    posterior_tags = []
    for k in posterior_unigrams:
        max_state = None
        max_p = float('-inf')
        for s, p in posterior_unigrams[k]:
            if p > max_p:
                max_state = s
                max_p = p
        posterior_tags.append(max_state)
    return posterior_tags


def split_train_file(trainfile):
    lines = open(trainfile, 'r').readlines()
    observations = []
    states = []
    for l in lines:
        if l.strip() != '':
            state = l.split('/')[1].strip()
            obs = l.split('/')[0].strip()
            states.append(state)
            observations.append(obs)
    return states, observations


def get_metrics(tags, answers, filter):
    correct_filtered_tags = 0.0
    total_filtered_tags = 0.0
    correct_filtered_indexes = [idx for idx, i in enumerate(filter) if (i == 1 and answers[idx] == tags[idx])]
    correct_filtered_tags += len(correct_filtered_indexes)
    total_filtered_tags += sum(filter)
    try:
        filtered_accuracy = 100 * correct_filtered_tags / float(total_filtered_tags)
    except ZeroDivisionError:
        filtered_accuracy = 0.0
    return filtered_accuracy


def get_metrics_bkup(tags, answers, known):
    correct_tags = 0
    total_tags = 0
    correct_known_tags = 0.0
    total_known_tags = 0.0
    correct_indexes = [i for i in range(len(tags)) if tags[i] == answers[i]]
    correct_tags += len(correct_indexes)
    total_tags += len(answers)
    correct_known_indexes = [idx for idx, i in enumerate(known) if (i == 1 and answers[idx] == tags[idx])]

    correct_known_tags += len(correct_known_indexes)
    total_known_tags += sum(known)
    known_accuracy = 100 * correct_known_tags / float(total_known_tags)
    try:
        unknown_accuracy = 100 * (correct_tags - correct_known_tags) / float(total_tags - total_known_tags)
    except ZeroDivisionError:
        unknown_accuracy = 0.0
    tagging_accuracy = "%.2f" % (100 * float(correct_tags) / float(total_tags))
    return tagging_accuracy, known_accuracy, unknown_accuracy


if __name__ == "__main__":
    try:
        train_file = argv[1]
        test_file = argv[2]
        raw_file = argv[3]
    except:
        train_file = '../data/ictrain'
        test_file = '../data/ictest'
        raw_file = '../data/icraw'


    original_states, original_observations = split_train_file(train_file)
    make_initial_mle_estimates(original_states, original_observations)
    raw_obs = read_raw_sentences(raw_file)
    add_raw_obs_to_initial_mle_estimates(raw_obs)
    save_current_to_original()
    answer_tags, test_obs = read_test_sentences(test_file)

    full, known, seen, novel = get_known_indexes(test_obs)
    answer_tags = filter(lambda x: x != BOUNDRY_STATE, answer_tags)
    for i in range(11):
        print '* * * ITERATION', i, '* * *'
        '''
        Viterbi
        '''
        viterbi_predicted_tags, max_p, alpha_pi_unused = get_viterbi_and_forward(test_obs)
        num_sentences = len(viterbi_predicted_tags)
        viterbi_predicted_tags = filter(lambda x: x != BOUNDRY_STATE, viterbi_predicted_tags)
        num_sentences -= len(viterbi_predicted_tags) # number of sentences is the number of boundry words detected and removed
        perpexity = (exp(-max_p / float(len(viterbi_predicted_tags) + num_sentences)))
        tav = get_metrics(viterbi_predicted_tags, answer_tags, full)
        kav = get_metrics(viterbi_predicted_tags, answer_tags, known)
        sav = get_metrics(viterbi_predicted_tags, answer_tags, seen)
        nav = get_metrics(viterbi_predicted_tags, answer_tags, novel)
        viterbi_perplexity_per_word = "%.3f" % (perpexity)
        '''
        display
        '''
        stderr.write(str('Tagging accuracy (Viterbi decoding): ' + str("%.2f" % tav) + '%\t'))
        stderr.write(str('(known: ' + str("%.2f" % kav) + '%\t'))
        stderr.write(str('seen: ' + str("%.2f" % sav) + '%\t'))
        stderr.write(str('novel: ' + str("%.2f" % nav) + '%)\n'))
        stderr.write('Perplexity per Viterbi-tagged test word: ' + viterbi_perplexity_per_word)
        stderr.write('\n')

        '''
        Forward-backward
        '''
        viterbi_predicted_tags_unused, max_p_unused, alpha_pi = get_viterbi_and_forward(raw_obs)
        posterior_unigrams, posterior_bigrams_accumilation, posterior_obs_accumilation, S = get_backwards(raw_obs, alpha_pi)
        posterior_predicted_tags = posterior_unigram_decoding(posterior_unigrams)
        posterior_predicted_tags.insert(0, BOUNDRY_STATE)
        make_new_mle_estimates(posterior_bigrams_accumilation, posterior_obs_accumilation, False) # we make new mles with counts from f-b
        set_current_from_new()
        posterior_predicted_tags = filter(lambda x: x != BOUNDRY_STATE, posterior_predicted_tags)
        #tap, kap, uap = get_metrics(posterior_predicted_tags, answer_tags, known)
        perpexity = (exp(-S / float(len(posterior_predicted_tags) + num_sentences)))
        posterior_perplexity_per_word = "%.4f" % (perpexity)
        '''
        display
        '''
        #strs = '%1.e' % exp(S)
        stderr.write('Perplexity per untagged raw word: ' + posterior_perplexity_per_word + '\n')
        #stderr.write('p(observations): ' + strs)
        stderr.write('\n')


