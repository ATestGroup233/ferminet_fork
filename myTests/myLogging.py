

import logging
import os
import time


class myLogging:
    
    def __init__(self, filename=None):
        
        if not os.path.exists('./logs'):
            os.mkdir('logs')
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        format = logging.Formatter('%(message)s')
        time_format = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
        
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.DEBUG)
        console_log.setFormatter(format)
        logger.addHandler(console_log)
        
        if filename is None:
            filename = time.strftime("%Y-%m-%d_%H:%M:%S") + ".log"
        file_log = logging.FileHandler(filename=os.path.join("logs", filename))
        file_log.setLevel(logging.DEBUG)
        file_log.setFormatter(time_format)
        logger.addHandler(file_log)
        
        self.logger = logger
        
    def debug(self, *args):
        self.logger.debug(*args)
        
    def info(self, *args):
        self.logger.info(*args)

if __name__ == "__main__":
    logging = myLogging()
    logging.info("what's the f**k")
