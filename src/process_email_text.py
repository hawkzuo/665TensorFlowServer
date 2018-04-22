import os
import re

import nltk
from bs4 import BeautifulSoup, Comment
from nltk.corpus import stopwords
from nltk import word_tokenize

# Useful packages: matplotlib; networkx; numpy


# total ham emails :
# ONLY INBOX    ->  7000 + [1000]
# Total         ->  11000 + [1600]
# Selected spam emails:
# 4 configurations with Ratio: [1:1, 1:2, 1:3, 1:5]
# 2 configurations with Source: [Most recent, Random Sample]
# Total configuration combinations = 20

# Train-Test Split: Either 80-20 or 70-30



# This regex is used to remove undesired characters
basic_spam_regex = re.compile("X-Spam.*\n")

# This regex is used to filter out certain html remainders
title_html_regex = re.compile('<[\s]*/[\s]*t[\s]*i[\s]*t[\s]*')

# This regex is used to filter out pattern like '=[??]*'
equal_sign_regex = re.compile('[=%][a-zA-Z0-9_]{0,2}')

# degit regex
digit_regex = re.compile('[0-9]')

# Punction marks regex
punct_regex = re.compile('["&<>!|,:;/(){}\[\]]')

# Non Character regex
non_char_regex = re.compile('[^a-zA-Z0-9_]')

# These special characters should be removed on every occurrence
html_special_words = ['09', '20', 'c2', 'a0', '0a', '0d',
                      'nbsp', 'nbs']

# These words should not in the feature dictionary
words_to_remove = set(stopwords.words('english'))


def pretty_soup_text(soup):
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out
    # kill all comments
    comments = soup.findAll(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    # TODO: kill all special characters
    soup.prettify(formatter=lambda s: s.replace(u'\xa0', ' '))

    return soup.get_text().replace('=', '')


if __name__ == '__main__':

    #
    folder_prefix = 'data/inbox_only'

    total_emails = len(os.listdir(folder_prefix))-1
    print(total_emails)

    for test_id in range(total_emails):
        with open(folder_prefix + '/ma' + str(test_id) + '.txt', 'r') as f:

            mail_body = f.read()
            body_v1 = basic_spam_regex.sub('', mail_body)
            body_v2 = equal_sign_regex.sub('', body_v1)
            # body_v1 = re.sub(basic_spam_regex, '', mail_body)
            # body_v2 = re.sub(equal_sign_regex, '', body_v1)

            # Convert all chars into lower case
            # Has both benefits and drawbacks
            body_v3 = body_v2.lower()

            for pt in html_special_words:
                body_v3 = body_v3.replace(pt, '')

            # Obtain the URL links count of the email first: [Roughly]
            url_link_count = body_v3.count('http')

            # Deal with HTML leftovers
            possible_match = title_html_regex.search(body_v3)
            if possible_match:
                plain_text_part = body_v3[0:possible_match.span()[0]]
                html_remain_body = body_v3[possible_match.span()[0]: -1]

                bs_soup = BeautifulSoup(html_remain_body, "html5lib")
                a_soup_text = pretty_soup_text(bs_soup)
                body_v4 = plain_text_part + a_soup_text
                # print('perform shorten')
            else:
                body_v4 = body_v3

            body_v4 = body_v4.replace('\n', ' ')


            # Now apply nltk process
            a_tokens = re.split(r'[ \t\n]+', body_v4)

            # Seems this method doesn't work so well
            # a_tokens = word_tokenize(body_v4)

            # TODO: Filter out some specific types, called [Lemmatization]
            # A type: 26ec8822dddd => Use re, count # of numbers >=4 then remove
            # B type: ':',';','!'... => Use re matching such chars then remove
            # C type:
            # WordNet resources on Lemmatization: change women to woman for example
            a_filtered_tokens = []
            for token in a_tokens:
                filtered_token = punct_regex.sub('', token)

                if len(filtered_token) < 1 or filtered_token.find('http') >= 0 or len(filtered_token) > 22:
                    continue
                elif len(filtered_token) == 1 and non_char_regex.search(filtered_token):
                    continue

                digits_count = len(digit_regex.findall(filtered_token))
                if 4 <= digits_count < len(filtered_token):
                    continue

                if filtered_token in words_to_remove:
                    continue

                if filtered_token[-1] == '.':
                    filtered_token = filtered_token[0:-1]

                a_filtered_tokens.append(filtered_token)

            wnl = nltk.WordNetLemmatizer()

            final_tokens = [wnl.lemmatize(t) for t in a_filtered_tokens]

            pass


# String
# Method	Functionality
# s.find(t)	index of first instance of string t inside s (-1 if not found)
# s.rfind(t)	index of last instance of string t inside s (-1 if not found)
# s.index(t)	like s.find(t) except it raises ValueError if not found
# s.rindex(t)	like s.rfind(t) except it raises ValueError if not found
# s.join(text)	combine the words of the text into a string using s as the glue
# s.split(t)	split s into a list wherever a t is found (whitespace by default)
# s.splitlines()	split s into a list of strings, one per line
# s.lower()	a lowercased version of the string s
# s.upper()	an uppercased version of the string s
# s.title()	a titlecased version of the string s
# s.strip()	a copy of s without leading or trailing whitespace
# s.replace(t, u)	replace instances of t with u inside s

# Regular Expression
# Operator	Behavior
# .	Wildcard, matches any character
# ^abc	Matches some pattern abc at the start of a string
# abc$	Matches some pattern abc at the end of a string
# [abc]	Matches one of a set of characters
# [A-Z0-9]	Matches one of a range of characters
# ed|ing|s	Matches one of the specified strings (disjunction)
# *	Zero or more of previous item, e.g. a*, [a-z]* (also known as Kleene Closure)
# +	One or more of previous item, e.g. a+, [a-z]+
# ?	Zero or one of the previous item (i.e. optional), e.g. a?, [a-z]?
# {n}	Exactly n repeats where n is a non-negative integer
# {n,}	At least n repeats
# {,n}	No more than n repeats
# {m,n}	At least m and no more than n repeats
# a(b|c)+	Parentheses that indicate the scope of the operators
# Symbols
# Symbol	Function
# \b	Word boundary (zero width)
# \d	Any decimal digit (equivalent to [0-9])
# \D	Any non-digit character (equivalent to [^0-9])
# \s	Any whitespace character (equivalent to [ \t\n\r\f\v])
# \S	Any non-whitespace character (equivalent to [^ \t\n\r\f\v])
# \w	Any alphanumeric character (equivalent to [a-zA-Z0-9_])
# \W	Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])
# \t	The tab character
# \n	The newline character



    pass
