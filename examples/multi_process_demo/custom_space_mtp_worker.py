from xbbo.utils.message_queue.worker import Worker

def custom_black_box_func(config):
    '''
    define black box function:
    y = x^2
    '''
    return config['x'] ** 2

if __name__ == "__main__":
    # Must run master first!
    # ---- Begin Worker ----
    worker = Worker(custom_black_box_func, '127.0.0.1', 5678, authkey=b'abc')
    worker.run()
