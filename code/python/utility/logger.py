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
