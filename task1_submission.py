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

import nltk, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics
import numpy as np

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

def find_chapters(book_file):
	with codecs.open(book_file, 'r', 'utf-8', errors='replace') as f:
		book_text = f.read()

	book_text = re.sub(r'\r\n', '\n', book_text)

	book_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', book_text)

	# book_text = re.sub(r'(?<!\n)\n\n(?!\n)', '  ', book_text)
	
	# with codecs.open(book_file.replace('.txt','_.txt'), 'w', 'utf-8', errors='replace') as f:
	# 	f.write(book_text)

	section_regex = re.compile(r'\n{3,}', re.MULTILINE)
	
	prefix_regex = re.compile(r'(?i)(?P<prefix_type>book|part|volume)\s+(the\s*)?(?P<prefix_number>\b[a-z]+\b|\b\d+\b)(?:\s*[:\-,.?!;]+\s+.*|\s+.*)')

	bookTitle_regex = re.compile(r'[A_Z0_9\s]+$')

	chapter_regex = re.compile(r'(?i)^(the)?\s*(chapter|chap)?\s+(the\s*)?(?P<chapter_number>\d+|[IVXLCDM]+)(?:\s*[:\-,.?!;]+\s*|\s*)(?P<chapter_title_prefix>\n\n)?(?P<chapter_content>.*)$',re.MULTILINE)

	chapterTitle_regex = re.compile(r'(?P<chapter_title>.*)$')
	
	chapters = {}
	book_prefix = ''
	volume_prefix = ''
	part_prefix = ''
	for section in section_regex.split(book_text):
		# bookTitle = bookTitle_regex.match(section)
		prefix = prefix_regex.match(section)
		chapter = chapter_regex.match(section)

		if prefix :
			type = prefix.group('prefix_type').lower()
			if type == 'book' :
				book_prefix = f"({prefix.group('prefix_type')} {prefix.group('prefix_number')}) "
			elif type == 'part':
				part_prefix = f"({prefix.group('prefix_type')} {prefix.group('prefix_number')}) "
			elif type == 'volume':
				volume_prefix = f"({prefix.group('prefix_type')} {prefix.group('prefix_number')}) "
			# else:
			# 	book_prefix = f"({bookTitle.group().strip})"
		if chapter:
			chapter_content = chapterTitle_regex.match(chapter.group('chapter_content').strip())
			chapter_content_prefix = chapter.group('chapter_title_prefix')
			if chapter_content_prefix and chapter_content.isupper():
				chapter_number = chapter.group('chapter_number')
				chapter_content_title = chapter_content.group('chapter_title').strip()
				full_prefix = book_prefix + volume_prefix + part_prefix
				chapters[full_prefix + chapter_number] = chapter_content_title
			if chapter_content_prefix and not chapter_content.isupper():
				chapter_number = chapter.group('chapter_number')
				chapter_content_title = ''
				full_prefix = book_prefix + volume_prefix + part_prefix
				chapters[full_prefix + chapter_number] = chapter_content_title
			else:
				chapter_number = chapter.group('chapter_number')
				chapter_content_title = chapter_content.group('chapter_title').strip()
				full_prefix = book_prefix + volume_prefix + part_prefix
				chapters[full_prefix + chapter_number] = chapter_content_title
	return chapters

def exec_regex_toc( file1_book, file2_book, file3_book) :

	# CHANGE BELOW CODE TO USE REGEX TO BUILD A TABLE OF CONTENTS FOR A BOOK (task 1)

	# Input >> www.gutenberg.org sourced plain text file for a whole book
	# Output >> toc.json = { <chapter_number_text> : <chapter_title_text> }

	# hardcoded output to show exactly what is expected to be serialized
	dictTOC1 = find_chapters(file1_book)
	dictTOC2 = find_chapters(file2_book)
	dictTOC3 = find_chapters(file3_book)

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	writeHandle = codecs.open( 'toc1.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictTOC1, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'toc2.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictTOC2, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

	writeHandle = codecs.open( 'toc3.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictTOC3, indent=2 )
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

	exec_regex_toc( book1_file, book2_file , book3_file )