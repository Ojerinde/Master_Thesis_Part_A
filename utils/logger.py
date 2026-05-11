"""
Logging utility — centralised logging configuration for all experiments.

Provides console and file logging with per-experiment log files,
timestamp formatting, and progress tracking.
"""

from config.paths import LOGS_DIR
import logging
import sys
from datetime import datetime


def setup_logger(
    name,
    log_file=None,
    level=logging.INFO,
    console_output=True,
    file_output=True
):
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (default: logs/{name}.log)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to file

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        # Create logs directory if it doesn't exist
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Default log file name
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = LOGS_DIR / f"{name.replace('.', '_')}_{timestamp}.log"
        elif isinstance(log_file, str):
            log_file = LOGS_DIR / log_file

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_experiment_logger(experiment_name, level=logging.INFO):
    """
    Get a logger for a specific experiment.

    Args:
        experiment_name: Name of the experiment (e.g., 'baseline_training')
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    log_file = LOGS_DIR / f"{experiment_name}.log"
    return setup_logger(
        name=f"experiment.{experiment_name}",
        log_file=log_file,
        level=level
    )


class ProgressLogger:
    """
    Simple progress logger for tracking experiment progress.
    """

    def __init__(self, total_steps, logger=None, desc="Progress"):
        """
        Initialize progress logger.

        Args:
            total_steps: Total number of steps
            logger: Logger instance (creates new if None)
            desc: Description of the progress
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.desc = desc

        if logger is None:
            self.logger = setup_logger(
                'progress', console_output=True, file_output=False)
        else:
            self.logger = logger

    def update(self, step=1, message=None):
        """
        Update progress.

        Args:
            step: Number of steps to advance (default 1)
            message: Optional message to display
        """
        self.current_step += step
        progress_pct = (self.current_step / self.total_steps) * 100

        base_msg = f"{self.desc}: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%)"

        if message:
            full_msg = f"{base_msg} - {message}"
        else:
            full_msg = base_msg

        self.logger.info(full_msg)

    def finish(self, message="Complete"):
        """Mark progress as complete."""
        self.logger.info(f"{self.desc}: {message}")


class ExperimentLogger:
    """
    Logger wrapper for experiment tracking with metrics.
    """

    def __init__(self, experiment_name, level=logging.INFO):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            level: Logging level
        """
        self.logger = get_experiment_logger(experiment_name, level)
        self.experiment_name = experiment_name
        self.start_time = None
        self.metrics = {}

    def start(self):
        """Start experiment timing."""
        self.start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info(f"Starting Experiment: {self.experiment_name}")
        self.logger.info(
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)

    def log_config(self, config):
        """
        Log experiment configuration.

        Args:
            config: Dictionary of configuration parameters
        """
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_metric(self, name, value, step=None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if step is not None:
            msg = f"[Step {step}] {name}: {value}"
        else:
            msg = f"{name}: {value}"

        self.logger.info(msg)

        # Store metric
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'value': value, 'step': step})

    def log_metrics(self, metrics_dict, step=None):
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric name -> value
            step: Optional step/epoch number
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)

    def info(self, message):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log error message."""
        self.logger.error(message)

    def finish(self, status="SUCCESS"):
        """
        Finish experiment and log summary.

        Args:
            status: Experiment status (SUCCESS, FAILED, etc.)
        """
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None

        self.logger.info("="*70)
        self.logger.info(f"Experiment Status: {status}")
        if duration:
            self.logger.info(f"Duration: {duration}")
        self.logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)


# Example usage
if __name__ == "__main__":
    # Test basic logger
    logger = setup_logger('test', log_file='test.log')
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Test progress logger
    progress = ProgressLogger(total_steps=10, desc="Testing")
    for i in range(10):
        progress.update(message=f"Processing item {i+1}")
    progress.finish()

    # Test experiment logger
    exp_logger = ExperimentLogger('test_experiment')
    exp_logger.start()
    exp_logger.log_config({'learning_rate': 0.001, 'batch_size': 32})
    exp_logger.log_metric('accuracy', 0.95, step=1)
    exp_logger.log_metrics({'precision': 0.94, 'recall': 0.96}, step=1)
    exp_logger.finish()

    print("\n✓ Logger test complete! Check logs/ directory.")
