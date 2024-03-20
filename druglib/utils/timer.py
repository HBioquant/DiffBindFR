# Copyright (c) MDLDrugLib. All rights reserved.
import contextlib, logging, datetime
import time as Time
from time import time
from typing import Optional
import signal, threading, _thread
from contextlib import contextmanager
from druglib.alerts import TimerError, TimeoutError

class Timer:
    """
    A flexible Timer class.
    :Example:
    >>> import time
    >>> from druglib.utils import Timer
    >>> with Timer():# simulate a code block that will run for 1s by __enter__ and __exit__
    >>>     time.sleep(1)
    1.000
    >>> timer = Timer
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """
    def __init__(self,
                 start:bool=True,
                 print_tmpl:Optional[str]=None
                 ):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()


    def start(self):
        """
        Start the timer.
        """
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    @property
    def is_running(self):
        """
        bool function indicating that wether the timer is running.
        """
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def since_start(self):
        """
        Total time since the timer is started.
        Returns:
             Time in seconds float
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """
        Time since the last checking.
        Either:func:`since_start` or :func:`since_last_check` is a checking
        operation.
        Returns:
            Time in seconds float
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur

_g_timers = {}# global timers

def check_time(timer_id):
    """
    Add check points in a single line.

    This method is suitable for runing a task on a list of items. A timer will
        be registered when the medthod is called for the first time.
    E.g:
        >>> import time
        >>> from druglib.utils import Timer, check_time
        >>> for i in range(1, 10):
        >>>     # simulate a code block
        >>>     time.sleep(i)
        >>>     check_time('task1')
        2.000
        3.000
        4.000
        ...
        9.000
    Args:
        timer_id:str: Timer identifier
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()

# @contextmanager
# def time_limit(seconds):
#     def signal_handler(signum, frame):
#         raise TimeoutError(f"Timed out during {seconds}s.")
#
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutError("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = Time.perf_counter()
    yield
    toc = Time.perf_counter()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)

def to_date(s: str):
    return datetime.datetime(
        year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10])
    )