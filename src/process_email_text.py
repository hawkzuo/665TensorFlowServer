import os
import re

import nltk
from bs4 import BeautifulSoup, Comment
from nltk.corpus import stopwords


# total ham emails :
# ONLY INBOX    ->  7000 + [1000]
# Total         ->  11000 + [1600]


# This regex is used to remove undesired characters
basic_spam_regex = re.compile("X-Spam.*\n")

# This regex is used to filter out certain html remainders
title_html_regex = re.compile('<[\s]*/[\s]*t[\s]*i[\s]*t[\s]*')

# This regex is used to filter out pattern like '=[??]*'
equal_sign_regex = re.compile('[=%][a-zA-Z0-9_]{0,2}')

# These special characters should be removed on every occurrence
html_special_words = ['09', '20', 'c2', 'a0', '0a']

# These words should not in the feature dictionary
words_to_remove = stopwords.words('english')


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
            body_v1 = re.sub(basic_spam_regex, '', mail_body)
            body_v2 = re.sub(equal_sign_regex, '', body_v1)
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


            # Now apply nltk package
            tokens = body_v4.split(' ')
            if len(tokens) < 200:
                print(len(tokens))


    pass
