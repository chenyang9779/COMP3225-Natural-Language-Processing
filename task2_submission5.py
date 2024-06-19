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

def find_chapters(chapter_file):
	with codecs.open(chapter_file, 'r', 'utf-8', errors='replace') as f:
		chapter_text = f.read()

	chapter_text = re.sub(r'\r\n', '\n', chapter_text)

	chapter_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', chapter_text)
	section_regex = re.compile(r'\n+', re.MULTILINE)
	question_regex = re.compile(r'(?:[?.!;:\]})\'‘"“\-]+\s*)(?P<question>[\w\s,\'’”\-]+\?)')

	questions = []

	for section in section_regex.split(chapter_text):
		for question in question_regex.finditer(section):
			questions.append(question.group('question'))
	# for i in range(len(questions)):
	# 	questions[i] = re.sub('^(\'|"|“|‘)','',questions[i])
        
	# for match in quotation_regex.finditer(chapter_text):
	# 	chapter_text.replace(match.group(),'')
	# 	for question in question_regex.finditer(match.group('speech_text')):
	# 		questions.append(question.group())

	# for question in question_regex.finditer(chapter_text):
	# 	questions.append(question.group())

	return questions

def exec_regex_questions( file1_chapter, file2_chapter, file3_chapter ) :

	# CHANGE BELOW CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (task 2)

	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> questions.txt = plain text set of extracted questions. one line per question.

	# hardcoded output to show exactly what is expected to be serialized
	setQuestions1 = set(find_chapters(file1_chapter))
	setQuestions2 = set(find_chapters(file2_chapter))
	setQuestions3 = set(find_chapters(file3_chapter))

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	writeHandle = codecs.open( 'questions1.txt', 'w', 'utf-8', errors = 'replace' )
	for strQuestion in setQuestions1 :
		writeHandle.write( strQuestion + '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'questions2.txt', 'w', 'utf-8', errors = 'replace' )
	for strQuestion in setQuestions2 :
		writeHandle.write( strQuestion + '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'questions3.txt', 'w', 'utf-8', errors = 'replace' )
	for strQuestion in setQuestions3 :
		writeHandle.write( strQuestion + '\n' )
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

	exec_regex_questions( chapter1_file, chapter2_file, chapter3_file )
