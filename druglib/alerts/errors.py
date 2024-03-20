# Copyright (c) MDLDrugLib. All rights reserved.

class Error(Exception):
    """Base class for exceptions."""

class TimeoutError(Error):
    def __init__(self, message):
        self.message = message
        super(TimeoutError, self).__init__(message)

class TimerError(Error):
    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)

class MultipleChainsError(Error):
    """An error indicating that multiple chains were found for a given ID."""

