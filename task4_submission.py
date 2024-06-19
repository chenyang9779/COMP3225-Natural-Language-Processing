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
		max_iterations=35,
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

def exec_ner( file1_chapter = None, file2_chapter = None, file3_chapter = None, ontonotes_file = None ) :

	# CHANGE CODE BELOW TO TRAIN A NER MODEL AND/OR USE REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (task 4)

	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> characters.txt = plain text set of extracted character names. one line per character name.

	# hardcoded output to show exactly what is expected to be serialized (you should change this)
	# only the allowed types for task 4 PERSON will be serialized
	train_sents = create_dataset( filePath = ontonotes_file)

	X_train = [extract_features(s) for s in train_sents]
	y_train = [extract_labels(s) for s in train_sents]	
	
	model = crf_model(X_train, y_train)

	file1 = preprocess_textfile(file1_chapter)
	file2 = preprocess_textfile(file2_chapter)
	file3 = preprocess_textfile(file3_chapter)

	dictNE1 = get_dict(model.predict(file1),file1)
	dictNE2 = get_dict(model.predict(file2),file2)
	dictNE3 = get_dict(model.predict(file3),file3)

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	# write out all PERSON entries for character list for subtask 4
	writeHandle = codecs.open( 'characters1.txt', 'w', 'utf-8', errors = 'replace' )
	if 'PERSON' in dictNE1 :
		for strNE in dictNE1['PERSON'] :
			strNE = re.sub(r'[\W]+$', '', strNE)
			strNE = re.sub(r'^[\W]+', '', strNE)
			strNE = strNE.strip()
			writeHandle.write( strNE.strip().lower()+ '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'characters2.txt', 'w', 'utf-8', errors = 'replace' )
	if 'PERSON' in dictNE2 :
		for strNE in dictNE2['PERSON'] :
			strNE = re.sub(r'[\W]+$', '', strNE)
			strNE = re.sub(r'^[\W]+', '', strNE)
			strNE = strNE.strip()
			writeHandle.write( strNE.strip().lower()+ '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'characters3.txt', 'w', 'utf-8', errors = 'replace' )
	if 'PERSON' in dictNE3 :
		for strNE in dictNE3['PERSON'] :
			strNE = re.sub(r'[\W]+$', '', strNE)
			strNE = re.sub(r'^[\W]+', '', strNE)
			strNE = strNE.strip()
			writeHandle.write( strNE.strip().lower()+ '\n' )
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