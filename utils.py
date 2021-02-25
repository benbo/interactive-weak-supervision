import numpy as np
import pandas as pd
import pickle
import re
import os
import sys
import torch
from torchmodels import TorchMLP
from snorkel.labeling.model import LabelModel
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer

AVAILABLEDATASETS = {'IMDB', 'Amazon', 'journalist_photographer', 'professor_physician', 'painter_architect',
                     'professor_teacher'}
BIASBIOSDSETS = {'journalist_photographer', 'professor_physician', 'painter_architect', 'professor_teacher'}
LFTYPES = {'unigram', 'uni_and_bi_grams'}
FEATURES = {'bow', 'tfidf', None, 'unibigram'}
REPLACE_NO_SPACE = re.compile(r"[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile(r"(<br\s*/><br\s*/>)|(-)|(/)")

ENGLISH_SENTIMENT_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "co", "con",
    "could", "de", "describe", "detail", "did", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "few", "fifteen", "fifty", "fill",
    "find", "first", "for", "former", "formerly", "forty",
    "found", "from", "front", "full", "further", "get", "give", "go", "got",
    "had", "has", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely",
    "nevertheless", "next",
    "now", "of", "off", "often", "on",
    "once", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "several", "she", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "through", "throughout",
    "thru", "thus", "to", "together", "top", "toward", "towards",
    "twelve", "twenty", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def generate_ngram_LFs(corpus, lftype, mindf=None, dname=None, labelmap=['negative sentiment', 'positive sentiment']):
    if lftype not in ['unigram', 'uni+biggram']:
        raise NotImplementedError

    if mindf is None:
        mindf = 20.0 / len(corpus)

    vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1),
                                 analyzer='word', max_df=0.90, min_df=mindf, max_features=None,
                                 vocabulary=None, binary=True)
    LFuni = vectorizer.fit_transform(corpus)
    vocablistunigrams = sorted([(value, key) for key, value in vectorizer.vocabulary_.items()], key=lambda x: x[0])
    vocablistunigrams = [x[1] for x in vocablistunigrams]
    LFunineg = LFuni.copy()
    LFunineg.data *= -1

    description = ['contains term: %s. LF vote: %s' % (word, labelmap[1]) for word in vocablistunigrams]
    description += ['contains term: %s. LF vote: %s' % (word, labelmap[0]) for word in vocablistunigrams]
    if lftype == 'unigram':
        return sparse.hstack((LFuni, LFunineg)), description
    else:

        if dname in BIASBIOSDSETS:
            stop_words = 'english'
        else:
            # maintain negations etc
            stop_words = ENGLISH_SENTIMENT_STOP_WORDS
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words=stop_words, ngram_range=(2, 2), analyzer='word',
                                     max_df=0.90, min_df=mindf, max_features=None, vocabulary=None, binary=True)
        LFbi = vectorizer.fit_transform(corpus)
        vocablistbigrams = sorted([(value, key) for key, value in vectorizer.vocabulary_.items()], key=lambda x: x[0])
        vocablistbigrams = [x[1] for x in vocablistbigrams]

        LFbineg = LFbi.copy()
        LFbineg.data *= -1

        descriptionunibi = ['contains term: %s. LF vote: %s' % (word, labelmap[1]) for word in vocablistunigrams]
        descriptionunibi += ['contains bigram: %s. LF vote: %s' % (word, labelmap[1]) for word in vocablistbigrams]
        descriptionunibi += ['contains term: %s. LF vote: %s' % (word, labelmap[0]) for word in vocablistunigrams]
        descriptionunibi += ['contains bigram: %s. LF vote:  %s' % (word, labelmap[0]) for word in vocablistbigrams]

        return sparse.hstack((LFuni, LFbi, LFunineg, LFbineg)), descriptionunibi


def get_final_set(mode, iwssession, npredict=100, r=None):
    """
        Function to create the final set of LFs we will use to model the
        latent class label Y. Returns final sets for all repeated runs of IWS.

        Parameters
        ----------
        mode : {'LSE a','LSE ac','AS'}
            mode to determine final set.
            AS: only uses LFs validated by user.
            LSE a: uses all LFs predicted to have good accuracy>r.
            LSE ac: uses a limited size set of LFs predicted to have good accuracy>r.
        iwssession : Object
            An instance of InteractiveWeakSupervision class
        npredict : int, default = 100
            Number of unverified LFs to add to the final set in LF ac mode
        r : float, default = None
            A threshold value in [0.5, 1]. If none, the r value in iwssession is used

        Returns
        -------
        LFsets: dict
            A dictonary containing, for each run of IWS, final LF indices for each iteration
            {runindex: {iteration : LF_indices})}
    """

    if mode not in {'LSE a', 'LSE ac', 'AS'}:
        raise ValueError('Choose one of: LSE a, LSE ac, AS')

    if mode == 'AS':
        if iwssession.acquisition != mode:
            raise ValueError('mode AS selected, but IWS was not run with AS acquisition')
    else:
        if iwssession.acquisition != 'LSE':
            raise ValueError('LSE mode selected but IWS was not run with LSE acquisition')

    if r is None:
        r = iwssession.straddle_threshold
    else:
        if not 0.5 <= r <= 1:
            raise ValueError('Choose r in [0.5,1.0]')

    # how many LFs in labelsequence are from initialization
    ninit = iwssession.nrandom_init + len(iwssession.initial_labels.keys())

    LFsets = {}
    coverage = None
    if mode == 'LSE ac':
        coverage = np.asarray((iwssession.LFs != 0).sum(0)).flatten() / iwssession.LFs.shape[0]

    for runidx, vals in iwssession.rawdatadict.items():
        LFsets[runidx] = {}
        labelvector, labelsequence, rawdata, timing, weightvector = vals

        itermax = len(labelsequence) - ninit
        for iteration in range(itermax):
            # get LF indices that we have responses for
            LFidxs = labelsequence[:iteration + ninit + 1]

            # First, grab all LFs that user indicated are good
            # If we are in AS mode, we only use these LFs
            traindxs = [x for x in LFidxs if labelvector[x] == 1]

            # establish LFs predicted to be good
            if mode == 'LSE a':
                pred_mean, testidxs = rawdata[iteration]
                # take all LFs predicted to be >= r
                traindxs += testidxs[pred_mean >= r].tolist()
            elif mode == 'LSE ac':
                pred_mean, testidxs = rawdata[iteration]
                pred_bool = pred_mean >= r
                if np.any(pred_bool):
                    # get coverage, test indices, predictions for LFs predicted to 
                    # have accuracy >=r
                    testcoverage = coverage[testidxs]
                    testcoveragetmp = testcoverage[pred_bool]
                    testidxtmp = testidxs[pred_bool]
                    preds = pred_mean[pred_bool]

                    # sort LFs, grab top npredict
                    sortval = np.multiply((2 * preds - 1), testcoveragetmp)
                    idxselect = np.argsort(sortval)[::-1][:npredict]
                    traindxs += testidxtmp[idxselect].tolist()

            LFsets[runidx][iteration] = traindxs

    return LFsets


def get_probabilistic_labels(iwssession, lfsets, device='cuda', gap=20,
                             class_balance=None, uniform=False, verbose=False):
    """
    Function to fit label model, train downstream classifier, and return test set predictions

    Parameters
    ----------
    iwssession : Object
        An instance of InteractiveWeakSupervision class
    lfsets : dict
         A dictonary containing, for each run of IWS, final LF indices for each iteration
            {runindex: {iteration : LF_indices})}
    device : str, default = 'cuda'
        String passed to torch to identify which device to use, e.g. "cpu" or 'cuda:0'
    gap : int, default = 20
        Provide downstream results every "gap" iterations
    class_balance : tuple, default = None
        Class balance tuple (negative class fraction, positive class fraction)
        passed to graphical model, e.g. class_balance = (0.5,0.5)
    uniform : bool, default = False
        Use uniform weighted LFs to obtain label instead of fitting a graphical model to learn weights.
    verbose : bool, default = False
        Print iteration info if true.

    Returns
    -------
    results: dict
        A dictionary containing the probabilistic train labels and a boolean filter index variable for each iws run,
        and each internal iteration. The filter index variable is True for every sample where we have at least one
        non-abstain vote.
        {runindex: {iteration_idx : (prob_labels,filteridx)})}
    """

    results = {}
    # for each run of IWS
    for key, iterdict in lfsets.items():
        results[key] = {}
        # establish which IWS iterations to obtain results for
        itermax = len(iterdict.keys())
        finaliter = itermax - 1
        iters_to_run = list(range(0, itermax, gap))
        if finaliter not in iters_to_run:
            # always obtain results for final iteration
            iters_to_run.append(finaliter)

        for iteration_idx in iters_to_run:
            if verbose:
                print('IWS run: %d' % key, ' iteration: %d' % iteration_idx)
            trainidxs = iterdict[iteration_idx]
            # get seleted LFs

            if uniform:
                LFStmp = iwssession.LFs_csc[:, trainidxs].copy()
                n, m = LFStmp.shape
                weights = np.ones(m)
                rowsums = np.asarray((LFStmp != 0).sum(1)).flatten()
                filteridx = rowsums != 0

                posevidence = ((LFStmp == 1).astype(np.float32)).dot(weights)
                negevidence = ((LFStmp == -1).astype(np.float32)).dot(weights)
                posevidence = np.asarray(posevidence).flatten()
                negevidence = np.asarray(negevidence).flatten()

                posevidence = np.clip(posevidence, 0.0, 700.0)
                negevidence = np.clip(negevidence, 0.0, 700.0)

                bin_posterior = np.exp(posevidence) / (np.exp(posevidence) + np.exp(negevidence))
                bin_posterior = bin_posterior.astype(np.float32)
            else:
                Lambdas = np.asarray(iwssession.LFs_csc[:, trainidxs].todense())
                # create snorkel LF format
                rowsums = (Lambdas != 0).sum(1)
                filteridx = rowsums != 0
                Lambda_snorkel = np.copy(Lambdas)
                Lambda_snorkel[Lambda_snorkel == 0] = -10
                Lambda_snorkel[Lambda_snorkel == -1] = 0
                Lambda_snorkel[Lambda_snorkel == -10] = -1

                # create variable to filter out samples with 0 LF votes

                # train label model
                if 'cuda' in device:
                    torch.cuda.empty_cache()
                    label_model = LabelModel(cardinality=2, verbose=True, device=device)
                    label_model.fit(Lambda_snorkel[filteridx], class_balance=class_balance)
                    torch.cuda.empty_cache()
                else:
                    label_model = LabelModel(cardinality=2, verbose=True)
                    label_model.fit(Lambda_snorkel[filteridx], class_balance=class_balance)

                # get label estimate
                posterior = label_model.predict_proba(Lambda_snorkel)
                bin_posterior = posterior[:, 1].astype(np.float32)

            tmpindicator = np.isnan(bin_posterior)
            if tmpindicator.sum() > 0:
                bin_posterior[tmpindicator] = np.median(bin_posterior[~tmpindicator])

            results[key][iteration_idx] = (bin_posterior, filteridx)
    return results


def end_classifier_from_prob_labels(Xtrain, Xtest, prob_label_dict, device='cuda',
                                    modelparams=None, verbose=False):
    """
    Function to fit label model, train downstream classifier, and return test set predictions

    Parameters
    ----------
    Xtrain : ndarray of shape (n training samples,d features)
        Features for training data
    Xtest : ndarray of shape (n test samples,d features)
        Features for test data
    prob_label_dict : dict
        Dictionary containing probabilistic labels obtained from get_probabilistic_labels
    device : str, default = 'cuda'
        String passed to torch to identify which device to use, e.g. "cpu" or 'cuda:0'
    class_balance : tuple, default = None
        Class balance tuple (negative class fraction, positive class fraction)
        passed to graphical model, e.g. class_balance = (0.5,0.5)
    modelparams : dict, defgault = None
        Dictionary containing sizes of hidden layers and activation functions of the downstream MLP
    verbose : bool, default = False
        Print iteration info if true.

    Returns
    -------
    results: dict
        A dictionary containing test set predictions
        {runindex: {iteration_idx : test_predictions})}
    """
    if modelparams is None:
        modelparams = {
            'h_sizes': [Xtrain.shape[1], 20, 20],
            'activations': [torch.nn.ReLU(), torch.nn.ReLU()]
        }


    # for each run of IWS
    results = {}
    for key, iterdict in prob_label_dict.items():
        results[key] = {}
        for iteration_idx, tup in iterdict.items():
            bin_posterior, filteridx = tup
            if verbose:
                print('IWS run: %d' % key, ' iteration: %d' % iteration_idx)

            # train classifier on label estimate and get test set prediction
            Xtrain_filtered = Xtrain[filteridx]
            probs_train_filtered = bin_posterior[filteridx]
            torch.cuda.empty_cache()
            model = TorchMLP(h_sizes=modelparams['h_sizes'], activations=modelparams['activations'],
                             optimizer='Adam', nepochs=250)

            if 'cuda' in device:
                tdevice = torch.device(device)
                model.model.to(tdevice)
                model.fit(Xtrain_filtered, probs_train_filtered, device=tdevice)
                test_predictions = model.predict_proba(Xtest, device=tdevice)
            else:
                model.fit(Xtrain_filtered, probs_train_filtered)
                test_predictions = model.predict_proba(Xtest)
            results[key][iteration_idx] = test_predictions
    return results

def train_end_classifier(Xtrain, Xtest, iwssession, lfsets, device='cuda', gap=20,
                         class_balance=None, modelparams=None, uniform=False, verbose=False):
    """
    Function to fit label model, train downstream classifier, and return test set predictions

    Parameters
    ----------
    Xtrain : ndarray of shape (n training samples,d features)
        Features for training data
    Xtest : ndarray of shape (n test samples,d features)
        Features for test data
    iwssession : Object
        An instance of InteractiveWeakSupervision class
    lfsets : dict
         A dictonary containing, for each run of IWS, final LF indices for each iteration
            {runindex: {iteration : LF_indices})}
    device : str, default = 'cuda'
        String passed to torch to identify which device to use, e.g. "cpu" or 'cuda:0'
    gap : int, default = 20
        Provide downstream results every "gap" iterations
    class_balance : tuple, default = None
        Class balance tuple (negative class fraction, positive class fraction)
        passed to graphical model, e.g. class_balance = (0.5,0.5)
    modelparams : dict, defgault = None
        Dictionary containing sizes of hidden layers and activation functions of the downstream MLP
    uniform : bool, default = False
        Use uniform weighted LFs to obtain label instead of fitting a graphical model to learn weights.
    verbose : bool, default = False
        Print iteration info if true.

    Returns
    -------
    results: dict
        A dictionary containing the probabilistic test set predictions for each iws run,
        and each internal iteration
        {runindex: {iteration_idx : test_predictions})}
    """
    if modelparams is None:
        modelparams = {
            'h_sizes': [Xtrain.shape[1], 20, 20],
            'activations': [torch.nn.ReLU(), torch.nn.ReLU()]
        }

    results = {}
    # for each run of IWS
    for key, iterdict in lfsets.items():
        results[key] = {}
        # establish which IWS iterations to obtain results for
        itermax = len(iterdict.keys())
        finaliter = itermax - 1
        iters_to_run = list(range(0, itermax, gap))
        if finaliter not in iters_to_run:
            # always obtain results for final iteration
            iters_to_run.append(finaliter)

        for iteration_idx in iters_to_run:
            if verbose:
                print('IWS run: %d' % key, ' iteration: %d' % iteration_idx)
            trainidxs = iterdict[iteration_idx]
            # get seleted LFs

            if uniform:
                LFStmp = np.asarray(iwssession.LFs_csc[:, trainidxs].todense())
                n, m = LFStmp.shape
                weights = np.ones(m)
                rowsums = np.asarray((LFStmp != 0).sum(1)).flatten()
                filteridx = rowsums != 0

                posevidence = ((LFStmp == 1).astype(np.float32)).dot(weights)
                negevidence = ((LFStmp == -1).astype(np.float32)).dot(weights)

                posevidence = np.clip(posevidence, 0.0, 700.0)
                negevidence = np.clip(negevidence, 0.0, 700.0)

                bin_posterior = np.exp(posevidence) / (np.exp(posevidence) + np.exp(negevidence))
                bin_posterior = bin_posterior.astype(np.float32)
            else:
                Lambdas = np.asarray(iwssession.LFs_csc[:, trainidxs].todense())
                # create snorkel LF format
                rowsums = (Lambdas != 0).sum(1)
                filteridx = rowsums != 0
                Lambda_snorkel = np.copy(Lambdas)
                Lambda_snorkel[Lambda_snorkel == 0] = -10
                Lambda_snorkel[Lambda_snorkel == -1] = 0
                Lambda_snorkel[Lambda_snorkel == -10] = -1

                # create variable to filter out samples with 0 LF votes

                # train label model
                if 'cuda' in device:
                    torch.cuda.empty_cache()
                    label_model = LabelModel(cardinality=2, verbose=True, device=device)
                    label_model.fit(Lambda_snorkel[filteridx], class_balance=class_balance)
                    torch.cuda.empty_cache()
                else:
                    label_model = LabelModel(cardinality=2, verbose=True)
                    label_model.fit(Lambda_snorkel[filteridx], class_balance=class_balance)

                # get label estimate
                posterior = label_model.predict_proba(Lambda_snorkel)
                bin_posterior = posterior[:, 1].astype(np.float32)

            tmpindicator = np.isnan(bin_posterior)
            if tmpindicator.sum() > 0:
                bin_posterior[tmpindicator] = np.median(bin_posterior[~tmpindicator])

                # train classifier on label estimate and get test set prediction
            Xtrain_filtered = Xtrain[filteridx]
            probs_train_filtered = bin_posterior[filteridx]
            torch.cuda.empty_cache()
            model = TorchMLP(h_sizes=modelparams['h_sizes'], activations=modelparams['activations'],
                             optimizer='Adam', nepochs=250)

            if 'cuda' in device:
                tdevice = torch.device(device)
                model.model.to(tdevice)
                model.fit(Xtrain_filtered, probs_train_filtered, device=tdevice)
                test_predictions = model.predict_proba(Xtest, device=tdevice)
            else:
                model.fit(Xtrain_filtered, probs_train_filtered)
                test_predictions = model.predict_proba(Xtest)
            results[key][iteration_idx] = test_predictions
    return results


def print_progress(iteration, total, decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    frac = '%d/%d' % (iteration, total)
    sys.stdout.write('\r|%s|%s%s %s' % (bar, percents, '%', frac)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


class ProgressBar:
    def __init__(self, decimals=1, bar_length=100):
        self.decimals = str(decimals)
        self.bar_length = bar_length

    def update(self, iteration, total):
        str_format = "{0:." + self.decimals + "f}"
        percents = str_format.format(100.0 * (iteration / float(total)))
        filled_length = int(round(self.bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)
        frac = '%d/%d' % (iteration, total)
        return '|%s|%s%s %s' % (bar, percents, '%', frac)


def preprocess_reviews(reviews):
    """
    Function to clean review text
    """
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews


def evaluate_binary(X, Ytrue, verbose=False):
    """
    Compute metrics for all labeling functions
    given the true binary labels.
    """
    if isinstance(Ytrue, list):
        Ytrue = np.array(Ytrue)
    if 0 in Ytrue:
        Ytrue[Ytrue == 0] = -1

    isnan = np.isnan(Ytrue)
    if isnan.sum() > 0:
        if verbose:
            print('Handling unlabeled samples')
        # ignore unlabeled samples
        X = X.tocsr()[~isnan].tocoo()
        Ytrue = Ytrue[~isnan]

    numpos = np.count_nonzero(Ytrue > 0)
    numneg = np.count_nonzero(Ytrue < 0)

    n = X.shape[0]
    res = X.multiply((Ytrue.reshape(-1, 1) > 0).astype(np.float32))
    tp = (res > 0).sum(0)
    fn = (res < 0).sum(0)

    res = X.multiply(-(Ytrue.reshape(-1, 1) < 0).astype(np.float32))
    tn = (res > 0).sum(0)
    fp = (res < 0).sum(0)

    # fraction correct
    frac_correct = (tp + tn) / n
    frac_correct = np.asarray(frac_correct).flatten()

    # fraction incorrect
    frac_incorrect = (fp + fn) / n
    frac_incorrect = np.asarray(frac_incorrect).flatten()

    # (correct - incorrect) / n 
    frac_goodness = (tp + tn - fp - fn) / n
    frac_goodness = np.asarray(frac_goodness).flatten()

    # positives:recall 
    val = tp + fp
    val[val == 0] = 1.0
    precisionpos = tp / val
    recallpos = tp / numpos

    # negatives: specificity
    val = tn + fn
    val[val == 0] = 1.0
    precisionneg = tn / val
    recallneg = tn / numneg

    coverage = np.asarray((X != 0).sum(0)).flatten() / X.shape[0]

    precisionpos = np.asarray(precisionpos).flatten()
    recallpos = np.asarray(recallpos).flatten()
    precisionneg = np.asarray(precisionneg).flatten()
    recallneg = np.asarray(recallneg).flatten()

    lfsign = np.asarray(np.sign(X.sum(0))).flatten()
    positive_funcs = lfsign > 0
    negative_funcs = lfsign < 0
    precision = np.zeros(X.shape[1])
    precision[negative_funcs] = precisionneg[negative_funcs]
    precision[positive_funcs] = precisionpos[positive_funcs]
    recall = np.zeros(X.shape[1])
    recall[negative_funcs] = recallneg[negative_funcs]
    recall[positive_funcs] = recallpos[positive_funcs]

    return coverage, precision, recall, frac_correct, frac_incorrect, frac_goodness


def evaluate_complex_binary(X, Ytrue, verbose=False):
    """
    Compute metrics for all labeling functions
    given the true binary labels.

    LFs do not just have to output 1 of the labels
    """
    if isinstance(Ytrue, list):
        Ytrue = np.array(Ytrue)
    if 0 in Ytrue:
        Ytrue[Ytrue == 0] = -1

    isnan = np.isnan(Ytrue)
    if isnan.sum() > 0:
        if verbose:
            print('Handling unlabeled samples')
        # ignore unlabeled samples
        X = X.tocsr()[~isnan].tocoo()
        Ytrue = Ytrue[~isnan]

    numpos = np.count_nonzero(Ytrue > 0)
    numneg = np.count_nonzero(Ytrue < 0)

    n = X.shape[0]
    res = X.multiply((Ytrue.reshape(-1, 1) > 0).astype(np.float32))
    tp = (res > 0).sum(0)
    fn = (res < 0).sum(0)

    res = X.multiply(-(Ytrue.reshape(-1, 1) < 0).astype(np.float32))
    tn = (res > 0).sum(0)
    fp = (res < 0).sum(0)

    # fraction correct
    frac_correct = (tp + tn) / n
    frac_correct = np.asarray(frac_correct).flatten()

    # fraction incorrect
    frac_incorrect = (fp + fn) / n
    frac_incorrect = np.asarray(frac_incorrect).flatten()

    # (correct - incorrect) / n 
    frac_goodness = (tp + tn - fp - fn) / n
    frac_goodness = np.asarray(frac_goodness).flatten()

    # positives:recall 
    val = tp + fp
    val[val == 0] = 1.0
    precisionpos = tp / val
    recallpos = tp / numpos

    # negatives: specificity
    val = tn + fn
    val[val == 0] = 1.0
    precisionneg = tn / val
    recallneg = tn / numneg

    numvotes = np.asarray((X != 0).sum(0)).flatten()
    coverage = numvotes / X.shape[0]

    precisionpos = np.asarray(precisionpos).flatten()
    recallpos = np.asarray(recallpos).flatten()
    precisionneg = np.asarray(precisionneg).flatten()
    recallneg = np.asarray(recallneg).flatten()

    lfsign = np.asarray(np.sign(X.sum(0))).flatten()
    positive_funcs = lfsign > 0
    negative_funcs = lfsign < 0
    precision = np.zeros(X.shape[1])
    precision[negative_funcs] = precisionneg[negative_funcs]
    precision[positive_funcs] = precisionpos[positive_funcs]
    recall = np.zeros(X.shape[1])
    recall[negative_funcs] = recallneg[negative_funcs]
    recall[positive_funcs] = recallpos[positive_funcs]

    accuracy = (tp + tn) / numvotes
    accuracy = np.asarray(accuracy).flatten()

    return coverage, precision, recall, frac_correct, frac_incorrect, frac_goodness, accuracy


def eval_scores(y_true, y_scores, verbose=True):
    """
    Compute a number of metrics for predicted Y probabilitues
    and return them. Optionally print them in verbose mode.
    """
    auc = roc_auc_score(y_true, y_scores)
    logloss = log_loss(y_true, y_scores)
    brier = np.square(y_true - y_scores).mean()
    if verbose:
        print('Brier score:', brier)
        print('Cross entropy loss:', logloss)
        print('AUC', auc)
    return auc, logloss, brier
