#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:51:56 2018

@author: dileepn
"""
from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt

# Functions to read a book, count words, and obtain word stats
def count_words(text):
    text = text.lower()
    skips = [",", ".", ";", ":", "'", '"', "!", "?", "-"]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = {}
    for word in text.split(" "):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

def count_words_fast(text):
    text = text.lower()
    skips = [",", ".", ";", ":", "'", '"', "!", "?", "-"]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts

def read_book(title_path):
    with open(title_path, 'r', encoding = "utf8") as current_text:
        text = current_text.read()
        text = text.replace("\n", " ").replace("\r", " ")
    return text

def word_stats(word_counts):
    num_unique = len(word_counts)
    length = sum(word_counts.values())
    return (num_unique, length)

# Reading multiple files
            
stats = pd.DataFrame(columns = ("Language", "Author", "Title", "Length", \
                                "Unique"))
book_dir = "./Books"
title_num = 1
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + "/" + language):
        for title in os.listdir(book_dir + "/" + language + "/" + author):
            inputfile = book_dir + "/" + language + "/" + author + "/" + title
            text = read_book(inputfile)
            (num_uniq, length) = word_stats(count_words(text))
            stats.loc[title_num] = language, author.capitalize(), \
                title.replace(".txt", ""), length, num_uniq
            title_num += 1

# Generte a plot
plt.figure(figsize = (10, 10))

subset = stats[stats.Language == "English"]
plt.loglog(subset.Length, subset.Unique, 'o', label = "English", color = "crimson")
subset = stats[stats.Language == "German"]
plt.loglog(subset.Length, subset.Unique, 'o', label = "German", color = "orange")
subset = stats[stats.Language == "French"]
plt.loglog(subset.Length, subset.Unique, 'o', label = "French", color = "forestgreen")
subset = stats[stats.Language == "Portuguese"]
plt.loglog(subset.Length, subset.Unique, 'o', label = "Portuguese", color = "blueviolet")

plt.legend()
plt.xlabel("Book Length")
plt.ylabel("Number of unique words")
plt.savefig("bookStats.pdf")  