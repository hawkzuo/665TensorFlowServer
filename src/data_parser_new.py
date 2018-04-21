import base64
import binascii
import logging
import quopri
import os
from email.parser import Parser

# Handling some other charsets such as:
from time import sleep

non_readable_charsets = {'GB2312', '"gb2312',
                         'so-2022-j', 'iso-2022-jp', 'iso-2022-JP',
                         'shift_jis', 'Shift_JIS',
                         'big5', '"big5',
                         '"Windows-1251'}
direct_readable_charsets = {'UTF-8', 'utf-8', 'tf-', 'TF-', '"utf-8', ' "utf-8',
                            'us-ascii', '"us-ascii',
                            'SO-8859-', 'ISO-8859-1', 'so-8859-', 'iso-8859-1'}

logger = logging.getLogger('data_parser_new')


# Insights: only do UTF-8 encoding data
# Reason: Consistency ++ other charset help little between each other





# Parse a single payload's data
def parse_single_payload(payload, mail_no, ignore_error=True):
    error_flag = 'ignore' if ignore_error else 'strict'
    readable_string = ''

    step_plain_content = ''
    step_web_page_content = ''

    assigned_charset = None
    is_plain_text = False
    raw_content_type = payload.get('Content-Type')
    content_options = raw_content_type.split(';') if raw_content_type else []
    # Generate useful options here
    for op in content_options:
        if 'charset' in op:
            assigned_charset = op[10:-1]
        if 'text/' in op:
            if op == 'text/plain':
                is_plain_text = True

    if payload.get('Content-Transfer-Encoding') == 'base64' or payload.get('Content-Transfer-Encoding') == 'BASE64':
        if assigned_charset not in non_readable_charsets:
            readable_string = ''
            next_level_payloads = payload.get_payload()
            if type(next_level_payloads) is list:
                for i in range(len(next_level_payloads)):
                    try:
                        raw_bytes = base64.b64decode(next_level_payloads[i].get_payload())
                        readable_string += raw_bytes.decode('utf-8', errors=error_flag)
                    except binascii.Error:
                        print('binascii.Error occurred on mail', mail_no)
                    except ValueError:
                        # Meaning itself is UTF-8
                        readable_string += next_level_payloads[i].get_payload()
                    except TypeError:
                        readable_string = ''
            else:
                try:
                    raw_bytes = base64.b64decode(next_level_payloads)
                    readable_string = raw_bytes.decode('utf-8', errors=error_flag)
                except binascii.Error:
                    print('binascii.Error occurred on mail', mail_no)
                except ValueError:
                    # Meaning itself is UTF-8
                    readable_string += next_level_payloads

    elif payload.get('Content-Transfer-Encoding') == '8bit' \
            or payload.get('Content-Transfer-Encoding') == '7bit':
        if assigned_charset in direct_readable_charsets or assigned_charset is None:
            readable_string = payload.get_payload()
        elif assigned_charset in non_readable_charsets:
            pass
        else:
            raw_bytes = quopri.decodestring(payload.get_payload())
            readable_string = raw_bytes.decode('utf-8', errors=error_flag)

    if is_plain_text:
        step_plain_content += readable_string
    else:
        # Default is webpage
        step_web_page_content += readable_string

    return step_plain_content, step_web_page_content


# Call this by parse_email_contents('02','2018')
def parse_email_contents(month, year, save_to_file=False, verbose=False, ignore_error=True):
    print('Started processing for month', month, 'year', year, 'parsing requests')
    sleep(1)

    # data folder is outside the project scope
    # otherwise IDE will report error
    folder_prefix = '../../data/m' + str(year) + str(month)
    out_prefix = '../../dataout/m' + str(year) + str(month)

    total_emails = len(os.listdir(folder_prefix))
    if total_emails > 60000:
        total_emails = 60000
    total_valid_emails = 0

    parser = Parser()

    for mail_no in range(total_emails):
        whole_plain_content = ''
        whole_web_page_content = ''
        with open(folder_prefix + '/Mail' + str(mail_no) + '.txt', 'r') as f:
            try:
                msg = parser.parse(f)
            except UnicodeDecodeError:
                logger.info('[Cannot Parse UTF] on mail', mail_no)
                print('Mail', mail_no, 'cannot be viewed as UTF-8')
                continue
            # The best way to solve the payload problem is to Imp. a
            # recursive function, however, since the depth=2 case is actually
            # rare in practice, won't bother recursion
            if msg.is_multipart():
                for payload in msg.get_payload():
                    plain, web = parse_single_payload(payload, mail_no, ignore_error)
                    whole_plain_content += plain
                    whole_web_page_content += web
            else:
                plain, web = parse_single_payload(msg, mail_no, ignore_error)
                whole_plain_content += plain
                whole_web_page_content += web

            print('Mail', mail_no, 'parsed')
            if len(whole_web_page_content) > 10 or len(whole_plain_content) > 10:
                total_valid_emails += 1

                if save_to_file:
                    if len(whole_web_page_content) > 10:
                        with open(out_prefix + '/ParsedMail' + str(total_valid_emails) + '_Web.txt', 'w') as fo:
                            fo.write(whole_web_page_content)
                    if len(whole_plain_content) > 10:
                        with open(out_prefix + '/ParsedMail' + str(total_valid_emails) + '_Plain.txt', 'w') as fo:
                            fo.write(whole_plain_content)

            if verbose:
                print('web:\n', whole_web_page_content, '\nplain:\n', whole_plain_content)
                sleep(2)

    #
    print('\nFinished processing for month', month, 'year', year, 'parsing requests')
    print('Total mails:', total_emails)
    print('Total valid mails:', total_valid_emails)


if __name__ == '__main__':
    # parse_email_contents('01', '2016', verbose=False, save_to_file=True)
    # parse_email_contents('02', '2016', verbose=False)

    # parse_email_contents('01', '2018', verbose=False, save_to_file=True)
    # parse_email_contents('02', '2018', verbose=False, save_to_file=True)
    # parse_email_contents('03', '2018', verbose=False, save_to_file=True)
    # parse_email_contents('04', '2018', verbose=False, save_to_file=True)

    pass