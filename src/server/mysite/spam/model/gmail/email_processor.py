import os
import re

import pickle
from time import sleep

import nltk
from bs4 import BeautifulSoup, Comment
from nltk.corpus import stopwords
from nltk import word_tokenize

# Useful packages: matplotlib; networkx; numpy

# This script is used to parse well-formatted email text files
# into pickle files


DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/'

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
    try:
        soup.prettify(formatter=lambda s: s.replace(u'\xa0', ' '))
    except RecursionError:
        pass
    try:
        text_body = soup.get_text()
    except Exception:
        text_body = ''

    return text_body.replace('=', '')


# TODO: Filter out some other types, called [Lemmatization]
# A type: 26ec8822dddd => Use re, count # of numbers >=4 then remove
# B type: ':',';','!'... => Use re matching such chars then remove
# C type:
# WordNet resources on Lemmatization: change women to woman for example
def generate_filtered_tokens(raw_tokens):
    temp = []
    for token in raw_tokens:
        filtered = punct_regex.sub('', token)

        if len(filtered) < 1 or filtered.find('http') >= 0 or len(filtered) > 22:
            continue
        elif len(filtered) == 1 and non_char_regex.search(filtered):
            continue

        digits_count = len(digit_regex.findall(filtered))
        if 4 <= digits_count < len(filtered):
            continue

        if filtered in words_to_remove:
            continue

        if filtered[-1] == '.':
            filtered = filtered[0:-1]

        temp.append(filtered)

    wnl = nltk.WordNetLemmatizer()

    result = [wnl.lemmatize(t) for t in temp]

    return result


# Parse ham emails into pickles
def parse_ham_emails(in_path, out_path, total_emails):
    counter = 0

    for i in range(total_emails):
        counter += 1
        print('Mail NO:', counter)
        if counter < 1:
            continue

        with open(in_path + '/om' + str(i) + '.txt', 'r') as f:

            mail_body = f.read()
            body_v1 = basic_spam_regex.sub('', mail_body)
            body_v2 = equal_sign_regex.sub('', body_v1)
            # body_v1 = re.sub(basic_spam_regex, '', mail_body)
            # body_v2 = re.sub(equal_sign_regex, '', body_v1)

            # Convert all chars into lower case
            # Has both benefits and drawbacks
            # Benefit: Perform Normalization on data
            # Drawback: CHINA and china are two words in dictionary

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
            # Seems this method doesn't work so well
            # a_tokens = word_tokenize(body_v4)
            a_tokens = re.split(r'[ \t\n]+', body_v4)
            final_tokens = generate_filtered_tokens(a_tokens)

            result_dict = {'url': url_link_count, 'content': final_tokens}

            with open(out_path + '/m' + str(i) + '.pickle', 'wb') as fp:
                pickle.dump(result_dict, fp)

            pass


# Parse spam emails into pickles
def parse_spam_emails(year, month):
    print('Started processing for month', month, 'year', year, 'parsing requests')
    sleep(1)
    in_prefix = DATA_PREFIX + 'dataout/m' + str(year) + str(month)
    out_prefix = DATA_PREFIX + 'spamout/m' + str(year) + str(month)
    with open(in_prefix + '/total.txt', 'r') as ft:
        total = int(ft.read())

    print('Total:', total)

    plain_parts = set(os.listdir(in_prefix + '/p'))
    web_parts = set(os.listdir(in_prefix + '/w'))

    for i in range(1, total + 1):

        if i % 200 == 0:
            print('Progress', i, '/', total)

        step_mail_name = 'Mail' + str(i) + '.txt'

        step_tokens = []
        step_url_counts = 0

        if step_mail_name in plain_parts:
            with open(in_prefix + '/p/' + step_mail_name, 'r') as f:
                plain_v1 = basic_spam_regex.sub('', f.read())
                plain_v2 = equal_sign_regex.sub('', plain_v1)
                plain_v3 = plain_v2.lower()
                plain_url_counts = plain_v3.count('http')

                plain_v4 = plain_v3.replace('\n', ' ')
                plain_raw_tokens = re.split(r'[ \t\n]+', plain_v4)
                plain_tokens = generate_filtered_tokens(plain_raw_tokens)

                step_tokens += plain_tokens
                step_url_counts += plain_url_counts
                pass

        if step_mail_name in web_parts:
            # Require BS to remove the wrapping
            with open(in_prefix + '/w/' + step_mail_name, 'r') as f:
                soup = BeautifulSoup(f.read(), "html5lib")
                parsed_web_content = pretty_soup_text(soup)

                web_tokens, web_url_counts = generate_tokens_from_parsed_soup_text(parsed_web_content)
                # body_v1 = basic_spam_regex.sub('', parsed_web_content)
                # body_v2 = equal_sign_regex.sub('', body_v1)
                # body_v3 = body_v2.lower()
                # web_url_counts = body_v3.count('http')
                #
                # body_v4 = body_v3.replace('\n', ' ')
                # web_raw_tokens = re.split(r'[ \t\n]+', body_v4)
                # web_tokens = generate_filtered_tokens(web_raw_tokens)

                step_tokens += web_tokens
                step_url_counts += web_url_counts
                pass

        result_dict = {'url': step_url_counts, 'content': step_tokens}

        with open(out_prefix + '/m' + str(i) + '.pickle', 'wb') as fp:
            pickle.dump(result_dict, fp)

        sleep(0.002)
    print('Finished processing for month', month, 'year', year, 'parsing requests\n')
    pass


def generate_tokens_from_parsed_soup_text(soup_text):
    body_v1 = basic_spam_regex.sub('', soup_text)
    body_v2 = equal_sign_regex.sub('', body_v1)
    body_v3 = body_v2.lower()
    web_url_counts = body_v3.count('http')

    body_v4 = body_v3.replace('\n', ' ')
    web_raw_tokens = re.split(r'[ \t\n]+', body_v4)
    web_tokens = generate_filtered_tokens(web_raw_tokens)

    return web_tokens, web_url_counts


if __name__ == '__main__':
    #
    # ham_in_prefix = 'data/overall'
    # ham_out_prefix = 'data/parsed'
    #
    # ham_total = len(os.listdir(ham_in_prefix)) - 1
    # print(ham_total)
    # sleep(2)
    #
    # parse_ham_emails(ham_in_prefix, ham_out_prefix, ham_total)

    # Training Datasets
    # parse_spam_emails('2018', '04')
    # parse_spam_emails('2018', '03')
    # parse_spam_emails('2018', '02')
    # parse_spam_emails('2018', '01')
    # parse_spam_emails('2017', '12')
    # parse_spam_emails('2017', '11')
    pass

    # Test set for 2017
    # parse_spam_emails('2017', '10')
    # parse_spam_emails('2017', '09')
    # parse_spam_emails('2017', '08')
    # parse_spam_emails('2017', '07')
    # parse_spam_emails('2017', '06')
    # parse_spam_emails('2017', '05')
    # parse_spam_emails('2017', '04')
    # parse_spam_emails('2017', '03')
    # parse_spam_emails('2017', '02')
    # parse_spam_emails('2017', '01')
    pass

    # Test set for 2016
    # parse_spam_emails('2016', '12')
    # parse_spam_emails('2016', '11')
    # parse_spam_emails('2016', '10')
    # parse_spam_emails('2016', '09')
    # parse_spam_emails('2016', '08')
    # parse_spam_emails('2016', '07')
    # parse_spam_emails('2016', '06')
    # parse_spam_emails('2016', '05')
    # parse_spam_emails('2016', '04')
    # parse_spam_emails('2016', '03')
    # parse_spam_emails('2016', '02')
    # parse_spam_emails('2016', '01')
    pass

    # Test set for 2015
    # parse_spam_emails('2015', '12')
    # parse_spam_emails('2015', '11')
    # parse_spam_emails('2015', '10')
    # parse_spam_emails('2015', '09')
    # parse_spam_emails('2015', '08')
    # parse_spam_emails('2015', '07')
    # parse_spam_emails('2015', '06')
    # parse_spam_emails('2015', '05')
    # parse_spam_emails('2015', '04')
    # parse_spam_emails('2015', '03')
    # parse_spam_emails('2015', '02')
    # parse_spam_emails('2015', '01')
    pass

    # Test set for 2014
    # parse_spam_emails('2014', '12')
    # parse_spam_emails('2014', '11')
    # parse_spam_emails('2014', '10')
    # parse_spam_emails('2014', '09')
    # parse_spam_emails('2014', '08')
    # parse_spam_emails('2014', '07')
    # parse_spam_emails('2014', '06')
    # parse_spam_emails('2014', '05')
    # parse_spam_emails('2014', '04')
    # parse_spam_emails('2014', '03')
    # parse_spam_emails('2014', '02')
    # parse_spam_emails('2014', '01')
    pass

    # Test set for 2013
    # parse_spam_emails('2013', '12')
    # parse_spam_emails('2013', '11')
    # parse_spam_emails('2013', '10')
    # parse_spam_emails('2013', '09')
    # parse_spam_emails('2013', '08')
    # parse_spam_emails('2013', '07')
    # parse_spam_emails('2013', '06')
    # parse_spam_emails('2013', '05')
    # parse_spam_emails('2013', '04')
    # parse_spam_emails('2013', '03')
    # parse_spam_emails('2013', '02')
    # parse_spam_emails('2013', '01')
    pass



    print('Finished all requests')

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
