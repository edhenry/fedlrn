import codecs
import pickle


def pickle_string_to_obj(string: str):
    """Pickle string to object
    
    Arguments:
        s {str} -- String object to pickle
    
    Returns:
        [pickle] -- base64 encoded pickled object
    """
    unmarshal = pickle.loads(codecs.decode(string.encode(), "base64"))
    return unmarshal

def obj_to_pickle_string(x):
    """Pickle object to string
    
    Arguments:
        obj {object} -- 
    """
    return codecs.encode(pickle.dumps(x), "base64").decode()