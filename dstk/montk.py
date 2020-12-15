"""monitoring tool-kit"""
import smtplib
import os
from email.message import EmailMessage
import sys
import traceback
from pathlib import Path


def send_email(to: str, subject: str, body: str) -> None:
    """send a text email to a passed email address. the sender email address is sourced from an
    environment variable"""
    address = os.environ['EMAIL_ADDRESS']
    msg = EmailMessage()
    msg['From'] = address
    msg['To'] = to
    msg['Subject'] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(address, os.environ['EMAIL_PASSWORD'])
        smtp.send_message(msg)


def make_error_log() -> str:
    """if called once an exception has been raised This function returns a string error log (including type,
    msg and traceback)"""
    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)
    stack_trace = [f"File : {tr[0]} , Line : {tr[1]}, Func.Name : {tr[2]}, Message : {tr[3]}" for tr in trace_back]
    stack_trace = '\n\t'.join(stack_trace)
    error_log: str = (f"Exception type: {ex_type.__name__}\n"
                      f"Exception message: {ex_value}\n"
                      f"Stack trace:\n\t {stack_trace}")
    return error_log


def send_error_log_email(to: str) -> None:
    """If an exception is raised send an error log message to param email"""
    error_log = make_error_log()
    send_email(to=to, subject=f'Error in app: {Path(__file__).name}', body=error_log)
