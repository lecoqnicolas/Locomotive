import logging


class RequestCounter:
    def __init__(self):
        self.pos_count = 0
        self.neg_count = 0

    @property
    def request_count(self):
        return self.pos_count + self.neg_count


def get_callback_with_counter(counter: RequestCounter):
    def counter_callback(result, error):
        if error is not None:
            logging.error(f"Error from server: {str(error)}")
            counter.neg_count += 1
        else:
            for output in result.as_numpy("translation"):
                logging.info(output)
            counter.pos_count += 1
    return counter_callback
