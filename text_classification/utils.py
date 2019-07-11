def save_dict(dic, ofn, output0=True):
    """
    Save a dict into a txt file
    :param dic: dict to be saved
    :param ofn: save file path
    :param output0: whether to save item in the dict with value==0
    :return: None
    """
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in dic.keys():
            if output0 or dic[k] != 0:
                fout.write(str(k) + "\t" + str(dic[k]) + "\n")
              
                
def load_dict(fn, func=str):
    """
    load a dict from a .txt file
    :param fn: load file path
    :param func: what the values are to be transfered
    :return: a dictionary
    """
    dic = {}
    with open(fn, encoding="utf-8") as fin:
        for lv in (ll.split('\t', 1) for ll in fin.read().split('\n') if ll != ""):
            dic[lv[0]] = func(lv[1])
    return dic


def load_list(fn):
    """
    load a list from a .txt file
    :param fn: load file path
    :return: list
    """
    with open(fn, encoding="utf-8") as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st


def save_list(st, ofn):
    """
    Save a list into a .txt file
    :param st: list to be saved
    :param ofn: save file path
    :return: None
    """
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in st:
            fout.write(str(k) + "\n")
