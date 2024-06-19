#!/bin/bash
# memory limit in kilobytes (7340032 = 7G)
ulimit -m 7340032 -v 7340032
# run python code
python task4_submission.py ontonotes_parsed.json eval_book3.txt eval_chapter1.txt eval_book8.txt eval_chapter2.txt eval_book9.txt eval_chapter4.txt