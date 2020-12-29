#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import traceback

import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    reference: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/
    """

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# class ColoredFormatter(logging.Formatter):

#     BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#     #The background is set with 40 plus the number of the color, and the foreground with 30

#     #These are the sequences need to get colored ouput
#     RESET_SEQ = "\033[0m"
#     COLOR_SEQ = "\033[1;%dm"
#     BOLD_SEQ = "\033[1m"

#     def formatter_message(message, use_color = True):
#         if use_color:
#             message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
#         else:
#             message = message.replace("$RESET", "").replace("$BOLD", "")
#         return message

#     COLORS = {
#         'WARNING': YELLOW,
#         'INFO': WHITE,
#         'DEBUG': BLUE,
#         'CRITICAL': YELLOW,
#         'ERROR': RED
#     }

#     def __init__(self, msg, use_color = True):
#         logging.Formatter.__init__(self, msg)
#         self.use_color = use_color

#     def format(self, record):
#         levelname = record.levelname
#         if self.use_color and levelname in COLORS:
#             levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
#             record.levelname = levelname_color
#         return logging.Formatter.format(self, record)


class Logger:

    def __init__(self, name=None):
        # create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        # create formatter
        # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
        # add formatter to ch
        handler.setFormatter(CustomFormatter())

        # add ch to logger
        self.logger.addHandler(handler)

    # def check(self, condition, message=""):
    #     if not condition:
    #         self.logger.error("Failed condition: %s", message)

    # def check_lt(self, objL, objR, message=""):
    #     self.check(objL < objR, "{} < {}. {}".format(objL, objR, message))

    # def check_le(self, objL, objR, message=""):
    #     self.check(objL <= objR, "{} <= {}. {}".format(objL, objR, message))

    # def check_gt(self, objL, objR, message=""):
    #     self.check(objL > objR, "{} > {}. {}".format(objL, objR, message))

    # def check_ge(self, objL, objR, message=""):
    #     self.check(objL >= objR, "{} >= {}. {}".format(objL, objR, message))

    # def check_eq(self, objL, objR, message=""):
    #     self.check(objL == objR, "{} == {}. {}".format(objL, objR, message))

    # def check_ne(self, objL, objR, message=""):
    #     self.check(objL != objR, "{} != {}. {}".format(objL, objR, message))

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
        print("---traceback---")
        for line in traceback.format_stack():
            print(line.strip())

    def fatal(self, message):
        self.logger.critical(message)
