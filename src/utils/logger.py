"""
Created By: ishwor subedi
Date: 2024-08-22
"""

import logging
import logging.config
import yaml
import os


def setup_logger(logger_name, log_file):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    with open('log_params.yaml', 'r') as file:
        config = yaml.safe_load(file)

    for handler in config['handlers'].values():
        handler['filename'] = f'logs/{log_file}'

    logging.config.dictConfig(config)

    return logging.getLogger(logger_name)
