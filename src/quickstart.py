"""
Shows basic usage of the Gmail API.

Lists the user's Gmail labels.
"""
from __future__ import print_function

import quopri
from time import sleep

from googleapiclient import errors
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

import csv
import base64
import email
import os

from bs4 import BeautifulSoup, Comment
import nltk

# Email Types
PROMOTION_FLAG = 'CATEGORY_PROMOTIONS'
SOCIAL_FLAG = 'CATEGORY_SOCIAL'
INBOX_FLAG = 'INBOX'

# Charsets
readable_encodings = {'quoted-printable'}
readable_charsets = {'us-ascii', 'US-ASCII',
                     'utf-8', 'UTF-8',
                     'iso-8859-1', 'ISO-8859-1',
                     'ascii', 'ASCII'}

# Setup the Gmail API
SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
store = file.Storage('credentials.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
    creds = tools.run_flow(flow, store)
simple_service = build('gmail', 'v1', http=creds.authorize(Http()))


def ListMessagesMatchingQuery(service, user_id, query='', include_spam=False):
    """List all Messages of the user's mailbox matching the query.

    Args:
      service: Authorized Gmail API service instance.
      user_id: User's email address. The special value "me"
      can be used to indicate the authenticated user.
      query: String used to filter messages returned.
      Eg.- 'from:user@some_domain.com' for Messages from a particular sender.
      include_spam: include the spam/trash emails or not
      label_ids: Constrain mails matching certain labels only

    Returns:
      List of Messages that match the criteria of the query. Note that the
      returned list contains Message IDs, you must use get with the
      appropriate ID to get the details of a Message.
    """
    try:
        response = service.users().messages().list(userId=user_id,
                                                   q=query,
                                                   includeSpamTrash=include_spam).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, q=query,
                                                       pageToken=page_token,
                                                       includeSpamTrash=include_spam).execute()
            messages.extend(response['messages'])

        return messages
    except errors.HttpError as error:
        print('An error occurred: %s' % error)


def save_user_message_ids_to_csv_file():
    total_messages = ListMessagesMatchingQuery(service=simple_service, user_id='me')

    with open('test.csv', 'w', newline='') as f:
        fieldnames = ['id', 'threadId']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for msg in total_messages:
            writer.writerow(msg)

    print('finished')


def GetSingleMessageWithId(msg_id, service):
    try:
        response = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
        label_set = set(response['labelIds'])

        msg_bytes = base64.urlsafe_b64decode(response['raw'].encode('ASCII'))
        mime_msg = email.message_from_bytes(msg_bytes)
        return label_set, mime_msg
    except errors.HttpError as error:
        print('An error occurred: %s' % error)
        return {}, ''


def save_messages_to_disk(filename, service):
    out_prefix = '../../hamout'
    if len(os.listdir(out_prefix)) == 0:
        os.mkdir(out_prefix + '/ionly')
    inbox_only_count = 0
    all_count = 0

    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        counter = 0
        for row in reader:
            counter += 1
            if counter < 0+666:
                continue
            labels, message = GetSingleMessageWithId(row['id'], service)

            if row['id'] == '162cf0f6bca59aca':
                print('here')

            if len(labels) >= 1:
                # Parse the message to gain the plain contents first
                readable_txt, readable_web = parse_single_email_to_plain(message)
                readable_web.replace('=', '')

                atext = ''
                if len(readable_web) > 10:
                    try:
                        soup = BeautifulSoup(readable_web, "html5lib")
                    except Exception as e:
                        print(e)
                        sleep(6)
                        continue
                    atext = pretty_soup_text(soup)
                if PROMOTION_FLAG in labels or SOCIAL_FLAG in labels:
                    # save to hamout root folder
                    pass
                else:
                    # save to ionly folder
                    if len(readable_txt) + len(atext) > 40:
                        with open(out_prefix + '/' + row['id'] + '_p.txt', 'w') as fo:
                            fo.write(readable_txt + atext)
                    inbox_only_count += 1
                if len(readable_txt) + len(atext) > 40:
                    with open(out_prefix + '/ionly/' + row['id'] + '_p.txt', 'w') as fo:
                        fo.write(readable_txt + atext)
                all_count += 1
            print('all count:', all_count)
            sleep(0.1)


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


# Single Email Message Parser
def parse_single_email_to_plain(msg):
    web_string = ''
    text_string = ''

    if msg.is_multipart():
        # raise TypeError('Unexpected type', 'multipart')
        for part in msg.get_payload():
            step_text, step_web = parse_single_email_to_plain(part)
            text_string += step_text
            web_string += step_web
    else:
        content_type = msg.get('Content-Type').split(';')
        text_type = content_type[0].split('/')[1]

        charset_options = content_type[1].replace(' ', '').replace('"', '').split('=') if len(
            content_type) >= 2 else None
        charset = charset_options[1] if charset_options and 'charset' in charset_options[0] else None

        content_encoding = msg.get('Content-Transfer-Encoding')

        if content_encoding and content_encoding.lower() == 'base64':
            # Try decoders first
            raw_bytes = base64.b64decode(msg.get_payload())
            if text_type == 'html':
                web_string += raw_bytes.decode(charset or 'utf-8', errors='strict')
            elif text_type == 'plain':
                text_string += raw_bytes.decode(charset or 'utf-8', errors='strict')
        elif content_encoding and content_encoding.lower() == '7bit':
            try:
                raw_bytes = quopri.decodestring(msg.get_payload())
                if text_type == 'html':
                    web_string += raw_bytes.decode('utf-8', errors='ignore')
                elif text_type == 'plain':
                    text_string += raw_bytes.decode('utf-8', errors='ignore')
            except ValueError:
                if text_type == 'html':
                    web_string += msg.get_payload()
                elif text_type == 'plain':
                    text_string += msg.get_payload()
        elif content_encoding in readable_encodings or charset in readable_charsets:
            if text_type == 'html':
                web_string += msg.get_payload()
            elif text_type == 'plain':
                text_string += msg.get_payload()
        else:
            print('here')

    return text_string, web_string


# Next is parse <html> into raw txt


if __name__ == '__main__':
    # Gmail API sample usage
    # results = simple_service.users().labels().list(userId='me').execute()
    # labels = results.get('labels', [])
    # if not labels:
    #     print('No labels found.')
    # else:
    #     print('Labels:')
    #     for label in labels:
    #         print(label['name'])

    # Gain Message IDs
    # save_user_message_ids_to_csv_file()

    #
    save_messages_to_disk('test.csv', simple_service)

    pass
