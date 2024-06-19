# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/01/29
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

def get_dict(result_pred, text):
	result = []
	for i in range(len(result_pred)):

		conlltags = []

		for tg, word in zip(result_pred[i], text[i]):
			conlltags.append(((word['word'], word['postag'], tg)))

		for tree in nltk.chunk.conlltags2tree(conlltags):
			if type(tree) == nltk.tree.Tree:
				l = tree.label()
				s = " ".join([token for token, pos in tree.leaves()]).lower().strip()
				result.append((s, l))

	dic = {}
	for value, key in result:
		if key not in dic:
			dic.setdefault(key, []).append(value)
		elif value not in dic[key]:
			dic.setdefault(key, []).append(value)

	return dic

def parseFile(f, dict_ontonotes):
	list = []
	for str_file in f :
		for str_sent_index in dict_ontonotes[str_file] :
			# ignore sents with non-PENN POS tags
			if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue
			if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue

			list_entry = []

			# compute IOB tags for named entities (if any)
			ne_type_last = None
			for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
				strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
				strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
				ne_type = None
				if 'ne' in dict_ontonotes[str_file][str_sent_index] :
					dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
					if not 'parse_error' in dict_ne :
						for str_NEIndex in dict_ne :
							if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
								ne_type = dict_ne[str_NEIndex]['type']
								break
				if ne_type != None :
					if ne_type == ne_type_last :
						strIOB = 'I-' + ne_type
					else :
						strIOB = 'B-' + ne_type
				else :
					strIOB = 'O'
				ne_type_last = ne_type

				list_entry.append( ( strToken, strPOS, strIOB ) )

			list.append( list_entry )
	return list

def create_dataset(filePath = None, max_files = None) :
	dataset_file = filePath
	
	# load parsed ontonotes dataset
	readHandle = codecs.open( dataset_file, 'r', 'utf-8', errors = 'replace' )
	str_json = readHandle.read()
	readHandle.close()
	dict_ontonotes = json.loads( str_json )

	# make a training and test split
	list_files = list( dict_ontonotes.keys() )
	# if len(list_files) > max_files :
		# list_files = list_files[ :max_files ]
	nSplit = math.floor( len(list_files))
	list_train_files = list_files[:nSplit]
	# list_test_files = list_files[ nSplit : ]

	# sent = (tokens, pos, IOB_label)
	list_train = parseFile(list_train_files, dict_ontonotes)
	
	# list_test = []

	return list_train

def word2features(sent, i):

	# print(sent)
	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'word' : word,
		'postag': postag,

		# token shape
		'word.lower()': word.lower(),
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),

		# token suffix
		# 'word.suffix': word.lower()[-3:],

		# POS prefix
		# 'postag[:2]': postag[:2],
	}
	if i > 0:
		word_prev = sent[i-1][0]
		postag_prev = sent[i-1][1]
		features.update({
			'-1:word.lower()': word_prev.lower(),
			'-1:postag': postag_prev,
			'-1:word.lower()': word_prev.lower(),
			'-1:word.isupper()': word_prev.isupper(),
			'-1:word.istitle()': word_prev.istitle(),
			'-1:word.isdigit()': word_prev.isdigit(),
			# '-1:word.suffix': word_prev.lower()[-3:],
			# '-1:postag[:2]': postag_prev[:2],
		})
	else:
		features['BOS'] = True

	if i < len(sent)-1:
		word_next = sent[i+1][0]
		postag_next = sent[i+1][1]
		features.update({
			'+1:word.lower()': word_next.lower(),
			'+1:postag': postag_next,
			'+1:word.lower()': word_next.lower(),
			'+1:word.isupper()': word_next.isupper(),
			'+1:word.istitle()': word_next.istitle(),
			'+1:word.isdigit()': word_next.isdigit(),
			# '+1:word.suffix': word_next.lower()[-3:],
			# '+1:postag[:2]': postag_next[:2],
		})
	else:
		features['EOS'] = True

	return features

def extract_features(sentence):
	return [word2features(sentence, i) for i in range(len(sentence))]

def extract_labels(sentence):
	return [label for token, postag, label in sentence]

def crf_model(X_train, y_train):
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=0.1,
		c2=0.1,
		max_iterations=20,
		all_possible_transitions=True
		)
	
	crf.fit(X_train, y_train)

	return crf

def preprocess_textfile(filename):
	with open(filename, 'r', encoding='utf-8', errors='replace') as file:
			text = file.read()

		# Replace newline characters and tab characters with a single space
	# text = re.sub(r'[\r\n\t]+', ' ', text)

		# Tokenize the text into sentences
	sentences = nltk.sent_tokenize(text)
	tagged_words = []

		# Process each sentence
	for sentence in sentences:
			# Tokenize the sentence into words and tag each word
		words = nltk.word_tokenize(sentence)
		# print(words)
		tagged = nltk.pos_tag(words)
		# print(tagged)
		tagged_words.append(tagged)
		# print(tagged_words)


	featured = [extract_features(sent) for sent in tagged_words]
	# featured = []
	# for word in tagged_words:
	# 	print(word)
	# 	featured.append(extract_features(word))


	return featured

# def task5_train_crf_model( X_train, Y_train, max_iter, labels) :
# 	# randomized search to discover best parameters for CRF model
# 	crf = sklearn_crfsuite.CRF(
# 		algorithm='lbfgs', 
# 		max_iterations=max_iter, 
# 		all_possible_transitions=True
# 	)
# 	params_space = {
# 		'c1': scipy.stats.expon(scale=0.5),
# 		'c2': scipy.stats.expon(scale=0.05),
# 	}

# 	# optimize for micro F1 score
# 	f1_scorer = make_scorer( sklearn_crfsuite.metrics.flat_f1_score, average='weighted', labels=labels )

# 	logger.info( 'starting randomized search for hyperparameters' )
# 	n_folds = 2
# 	n_candidates = 10
	
# 	rs = sklearn.model_selection.RandomizedSearchCV(crf, params_space, cv=n_folds, verbose=1, n_jobs=-1, n_iter=n_candidates, scoring=f1_scorer)

	
# 	rs.fit(X_train, Y_train)
	

# 	# output the results
# 	logger.info( 'best params: {}'.format( rs.best_params_ ) )
# 	logger.info( 'best micro F1 score: {}'.format( rs.best_score_ ) )
# 	logger.info( 'model size: {:0.2f}M'.format( rs.best_estimator_.size_ / 1000000 ) )
# 	logger.info( 'cv_results_ = ' + repr(rs.cv_results_) )

# 	# visualize the results in hyperparameter space
# 	_x = [s['c1'] for s in rs.cv_results_['params']]
# 	_y = [s['c2'] for s in rs.cv_results_['params']]
# 	_c = [s for s in rs.cv_results_['mean_test_score']]

# 	# return the best model
# 	crf = rs.best_estimator_
# 	return crf

def exec_ner( file1_chapter = None, file2_chapter = None, file3_chapter = None, ontonotes_file = None ) :

	# CHANGE CODE BELOW TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (task 3)

	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }
	
	train_sents = create_dataset( filePath = ontonotes_file)

	X_train = [extract_features(s) for s in train_sents]
	y_train = [extract_labels(s) for s in train_sents]	
	
	# X_test = [extract_features(s) for s in test_sents]
	# Y_test = [extract_labels(s) for s in test_sents]

	# set_labels = set([])
	# for data in [y_train,Y_test] :
	# 	for n_sent in range(len(data)) :
	# 		for str_label in data[n_sent] :
	# 			set_labels.add( str_label )
	# labels = list( set_labels )

	# labels.remove('O')

	# model = task5_train_crf_model(X_train, y_train, 100, labels)
	# print(model)
	model = crf_model(X_train, y_train)

	file1 = preprocess_textfile(file1_chapter)
	file2 = preprocess_textfile(file2_chapter)
	file3 = preprocess_textfile(file3_chapter)

	dictNE1 = get_dict(model.predict(file1),file1)
	dictNE2 = get_dict(model.predict(file2),file2)
	dictNE3 = get_dict(model.predict(file3),file3)
	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	# FILTER NE dict by types required for task 3
	listAllowedTypes = [ 'DATE', 'CARDINAL', 'ORDINAL', 'NORP' ]
	
	
	listKeys = list( dictNE1.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE1[strKey])) :
			dictNE1[strKey][nIndex] = dictNE1[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE1[strKey]
#123
	listKeys = list( dictNE2.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE2[strKey])) :
			dictNE2[strKey][nIndex] = dictNE2[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE2[strKey]

	listKeys = list( dictNE3.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE3[strKey])) :
			dictNE3[strKey][nIndex] = dictNE3[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE3[strKey]

	# write filtered NE dict
	writeHandle = codecs.open( 'ne1.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictNE1, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'ne2.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictNE2, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'ne3.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictNE3, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

if __name__ == '__main__':
	if len(sys.argv) < 8 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book1_file = sys.argv[2]
	chapter1_file = sys.argv[3]
	book2_file = sys.argv[4]
	chapter2_file = sys.argv[5]
	book3_file = sys.argv[6]
	chapter3_file = sys.argv[7]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book1 = ' + repr(book1_file) )
	logger.info( 'chapter1 = ' + repr(chapter1_file) )
	logger.info( 'book2 = ' + repr(book2_file) )
	logger.info( 'chapter2 = ' + repr(chapter2_file) )
	logger.info( 'book3 = ' + repr(book3_file) )
	logger.info( 'chapter3 = ' + repr(chapter3_file) )

	# DO NOT CHANGE THE CODE IN THIS FUNCTION

	exec_ner( chapter1_file, chapter2_file, chapter3_file, ontonotes_file )

	logger.info( 'done')

