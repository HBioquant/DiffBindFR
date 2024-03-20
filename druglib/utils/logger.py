# Copyright (c) MDLDrugLib. All rights reserved. 
# Reference by https://github.com/open-mmlab/mmcv/blob/master/mmcv/mmcv/utils/logging.py
import os, argparse, warnings, logging
from abc import ABC, abstractmethod
from numbers import Number
from typing import Optional, Union, Dict, Callable, Tuple
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import torch.distributed as dist

try:
    from torch.utils.tensorboard import SummaryWriter
except AttributeError as e:
    warnings.warn(f"Maybe your setuptools version > 59.5.0, use tensorboardX instead.\n Error message: {e}")
    from tensorboardX import SummaryWriter

try:
    import wandb
except ImportError:
    warnings.warn("wandb library is needed for druglib. Please use `pip install wandb` to get it.")
    wandb = None

logger_initialized = {}

def get_logger(
    name:str = 'MDLDrugLib',
    log_file:Optional[str] = None,
    log_level:int = logging.INFO,
    io_mode:str = 'w'
) -> logging.Logger:
    """
    Setup a logger with your custom name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger
    will be directly returned. During initialization, a StreamHandler will 
    always be added. If `log_file` is specified and the process rank is 0, a
    FileHandler will be also added.

    Args:
        name:str, Optional, Defaults to MDLDrugLib: Logger name.
        log_file:str | None, Optional, Defaults to None: log file name. If specified, 
            a FileHandler will be added to the logger.
        log_level:int, Optional, Defaults to logging.INFO: The logger level. Note that only
            the process of rank 0 is affected, and other processes will set the level to "Error"
            thus be slient most of the time.
        io_mode:str, Optional, Defaults to "w": The file mode used in opening log file.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the 
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    
    # handle duplicate logs to the console
    # starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root level 
    # handle causes logging messages from rank > 0 processes to unexpectedly show up
    # on the console, creating much unwanted clutter.
    # To fix this issue, the solution is to set root logger's StreamHandler, if any, to
    # log at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)
    
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # The default mode of the official logger is `a`.
        # Thus, here we provide an interface to  change the
        # file mode to the default `w` mode.
        file_handler = logging.FileHandler(log_file, io_mode)
        handlers.append(file_handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    
    logger_initialized[name] = True

    return logger

def print_log(
    msg:str, 
    logger:Union[logging.Logger, str, None] = None,
    log_level:int = logging.INFO
) -> None:
    """
    Print a log message.
    Args:
        msg:str: The message to be logged.
        logger: logging.logger | str | None: The logger to be used.
            Some special loggers are:
                - "silent": no message will be printed.
                - "other str": the logger obtained with `get_root_logger(logger)`.
                - "None": The `print()` function will be used to print log messages.
        level:int, Optional, Defaults to logging.INFO: Logging level Only available when `logger`
            is a logger object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(log_level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(log_level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"slicent" or None, but got {type(logger)}'
        )

RL_LOG_DATA_TYPE = Dict[str, Union[int, Number, np.number, np.ndarray]]

class BaseRLLogger(ABC):
    """
    The base class for any logger which is compatible with trainer.
    Try to overwrite write() method to use your own writer.
    Args:
        train_interval:int: The log interval in log_train_data() method. Default  to 1000.
        test_interval:int: The log interval in log_test_data() method. Default to 1.
        update_interval:int: The log interval in log_update_date() method. Default to 1000.
    """
    def __init__(
            self,
            train_interval:int = 1000,
            test_interval:int = 1,
            update_interval:int = 1000,
    ) -> None:
        super().__init__()
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1

    @abstractmethod
    def write(
            self,
            step_type:str,
            step:int,
            data:RL_LOG_DATA_TYPE
    ) -> None:
        """
        Specify how the writer is used to log data.
        Args:
            step_type:str: Namespace which the data dict belongs to.
            step:int: Stands for the ordinate of the data dict.
            data:RL_LOG_DATA_TYPE: the data to write with dict-like-format ``{key: value}``.
        """
        pass

    def log_train_data(
            self,
            collect_results:dict,
            step:int,
    ) -> None:
        """
        Use writer to log statistics generated during training.
        Args:
            collect_results:dict: a dict containing information of data collected
                in training phase, i.e., from returns of collector.collect().
            step:int: stands for the timestep the collect_result being logged.
        """
        if collect_results["n/ep"] > 0:
            if (step - self.last_log_train_step) >= self.train_interval:
                log_data = {
                    "train/episode": collect_results["n/ep"],
                    "train/reward": collect_results["rew"],
                    "train/length": collect_results["len"],
                }
                self.write(
                    step_type = "train/env_step",
                    step = step,
                    data = log_data,
                )
                self.last_log_train_step = step

    def log_test_data(
            self,
            collect_results:dict,
            step:int,
    ) -> None:
        """
        Use writer to log statistic generated during inference.
        Args:
            collect_results:dict: a dict containing information of data collected
                in inference phase, i.e., from returns of collector.collect().
            step:int: stands for the timestep the collect_result being logged.
        """
        assert collect_results["n/ep"] > 0
        if (step - self.last_log_test_step) >= self.test_interval:
            log_data = {
                "inference/env_step":step,
                "inference/reward":collect_results["rew"],
                "inference/length":collect_results["len"],
                "inference/reward_std":collect_results["rew_std"],
                "inference/length_std":collect_results["len_std"],
            }
            self.write(
                step_type = "inference/env_step",
                step = step,
                data = log_data,
            )
            self.last_log_test_step = step

    def log_update_data(
            self,
            update_results:dict,
            step:int,
    ) -> None:
        """
        Use writer to log statistic generated during updating.
        Args:
            update_results:dict: a dict containing information of data collected
                in updating phase, i.e., from returns of policy.update().
            step:int: stands for the timestep the update_result being logged.
        """
        if (step - self.last_log_update_step) >= self.update_interval:
            log_data = {f"update/{k}":v for k, v in update_results.items()}
            self.write(
                step_type = "update/gradient_step",
                step = step,
                data = log_data
            )
            self.last_log_update_step = step

    def save_data(
            self,
            epoch:int,
            env_step:int,
            gradient_step:int,
            checkpoint_save_fn:Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        """
        Use writer to log metadata when calling `checkpoint_save_fn` in trainer.
        Args:
            epoch:int: The epoch in trainer.
            env_step:int: The env_step in trainer.
            gradient_step:int: The gradient_step in trainer.
            checkpoint_save_fn:Optional[Callable[[int, int, int], None]]: A hook defined by user.
        """
        pass

    def restore_data(self) -> Tuple[int, int, int]:
        """
        Return the metadata from existing log.
        If it finds nothing or an error occurs during the recover process, it
            will return the default parameters.
        Returns:
            epoch:int, env_step:int, gradient_step:int
        """
        pass

class RLLoggerPlaceholder(BaseRLLogger):
    """
    A logger that does nothing. Used as the placeholder in trainer.
    """
    def __init__(self) -> None:
        super(RLLoggerPlaceholder, self).__init__()

    def write(
            self,
            step_type:str,
            step:int,
            data:RL_LOG_DATA_TYPE
    ) -> None:
        """
        Note: The RLLoggerPlaceholder writes nothing.
        """
        pass

class TensorboardRLLogger(BaseRLLogger):
    """
    A logger that relies on tensorboard SummaryWriter by default to visualization
        and log statistics.
    Args:
        writer:SummaryWriter: A writer for log_data
        train_interval:int: The log interval in log_train_data() method. Default  to 1000.
        test_interval:int: The log interval in log_test_data() method. Default to 1.
        update_interval:int: The log interval in log_update_date() method. Default to 1000.
        save_interval:int: The log interval in save_data() method. Default to 1
            (save at the end of every epoch).
    """
    def __init__(
            self,
            writer:SummaryWriter,
            train_interval:int = 1000,
            test_interval:int = 1,
            update_interval:int = 1000,
            save_interval:int = 1,
    ):
        super(TensorboardRLLogger, self).__init__(train_interval, test_interval, update_interval)
        self.save_interval = save_interval
        self.last_save_step = -1
        self.writer = writer

    def write(
            self,
            step_type:str,
            step:int,
            data:RL_LOG_DATA_TYPE
    ) -> None:
        for k, v in data.items():
            self.writer.add_scalar(k, v, global_step = step)
        self.writer.flush()

    def save_data(
            self,
            epoch:int,
            env_step:int,
            gradient_step:int,
            checkpoint_save_fn:Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if checkpoint_save_fn and (epoch - self.last_save_step) >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_save_fn(epoch, env_step, gradient_step)
            self.write("save/epoch", epoch, {"save/epoch":epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write("save/gradient_step", gradient_step, {"save/gradient_step": gradient_step})

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:# epoch / gradient step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:# offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step

class WandbRLLogger(BaseRLLogger):
    """
    Weights and Biases logger that seeds data to https://wandb.ai/.
    This logger creates three panels with plots: train, test and update.
    Make sure to select the correct access for each panel in weights and biases:
        - `train/env_step` for train plots;
        - `test/env_step` for test plots;
        - `update/gradient_step` for update plots.
    Args:
        project:str: W&B project name. Default to "druglib".
        name:Optional[str]: W&B run name. Default to None. If None, random name is assigned.
        entity:Optional[str]: W&B team/organization name. Default to None.
        run_id:Optional[str]: Run id of W&B run to be resumed. Default to None.
        cfg:Optional[argparse.Namespace]: Namespace config, experient configurations. Default to None.
        train_interval:int: The log interval in log_train_data() method. Default  to 1000.
        test_interval:int: The log interval in log_test_data() method. Default to 1.
        update_interval:int: The log interval in log_update_date() method. Default to 1000.
        save_interval:int: The log interval in save_data() method. Default to 1
            (save at the end of every epoch).

    Example of usage:
        with wandb.init(project = "Your project"):
            logger = WandbRLLogger()
            result = onpolicy_trainer(policy, train_collector, test_collector, logger = logger)
    """
    def __init__(
            self,
            project:str = "druglib/RL",
            name:Optional[str] = None,
            entity:Optional[str] = None,
            run_id:Optional[str] = None,
            cfg:Optional[argparse.Namespace] = None,
            train_interval:int = 1000,
            test_interval:int = 1,
            update_interval:int = 1000,
            save_interval: int = 1,
    ):
        super(WandbRLLogger, self).__init__(train_interval, test_interval, update_interval)
        self.save_interval = save_interval
        self.last_save_step = -1
        self.restored = False

        self.wandb_run = wandb.init(
            project = project,
            name = name,
            entity = entity,
            id = run_id,
            config = cfg,
            resume = 'allow',
            monitor_gym = True,
        ) if not wandb.run else wandb.run
        self.wandb_run._label(repo = "druglib")

    def write(
            self,
            step_type:str,
            step:int,
            data:RL_LOG_DATA_TYPE
    ) -> None:
        data[step_type] = step
        wandb.log(data)

    def save_data(
            self,
            epoch:int,
            env_step:int,
            gradient_step:int,
            checkpoint_save_fn:Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if checkpoint_save_fn and (epoch - self.last_save_step) >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_pth = checkpoint_save_fn(epoch, env_step, gradient_step)
            checkpoint_artifact = wandb.Artifact(
                "run_" + self.wandb_run.id + "_checkpoint",
                type = 'model',
                metadata = {
                    "save/epoch":epoch,
                    "save/env_step":env_step,
                    "save/gradient_step":gradient_step,
                    "checkpoint_path":str(checkpoint_pth),
                }
            )
            checkpoint_artifact.add_file(str(checkpoint_pth))
            self.wandb_run.log_artifact(checkpoint_artifact)

    def restore_data(self) -> Tuple[int, int, int]:
        checkpoint_artifact = self.wandb_run.use_artifact(
            "run_" + self.wandb_run.id + "_checkpoint:latest"
        )
        assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist."
        checkpoint_artifact.download(
            os.path.dirname(checkpoint_artifact.metadata['checkpoint_path'])
        )

        try:# epoch / gradient step
            epoch = checkpoint_artifact.metadata["save/epoch"]
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:# offline trainer doesn't have env_step
            env_step = checkpoint_artifact.metadata["save/env_step"]
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0
        return epoch, env_step, gradient_step