import codecs
import pickle


def pickle_string_to_obj(string: str):
    """Pickle string to object
    
    Arguments:
        s {str} -- String object to pickle
    
    Returns:
        [pickle] -- base64 encoded pickled object
    """
    return pickle.loads(codecs.decode(string.encode(), "base64"))

def obj_to_pickle_string(obj: object):
    """Pickle object to string
    
    Arguments:
        obj {object} -- 
    """
    return codecs.encode(pickle.dumps(x), "base64").decode()
