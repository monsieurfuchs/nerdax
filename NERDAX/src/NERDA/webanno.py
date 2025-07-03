import pandas as pd
import simplejson as json
import re, os


def __webanno_to_df(filepath):
   """
   Convert a webanno tsv file into a dataframe.
   -----
   Params:
     filepath: string - the path to the csv file
   Returns:
     a dataframe object
   """
   df = pd.read_csv(
        filepath,
        sep='\t',
        usecols=[0,1,2,3,4,5],
        names=['sentence_token', 'char', 'token', 'label', 'rel', 'rel_src'],
        comment='#'
   )
   return df



def __sentence_id(sentence_token):
    """
    take a sentence-token id of the form dd-dd
    and return the sentence id
    ----
    Parameters:
       sentence-token: string - the sentence-token id 
    Returns:
       the sentence id as int
    """
    try:
       splitted = sentence_token.split('-')
    except:
       return None
    return int(splitted[0])
    
    
    
def __unescape(text):
    """
    parses a string and returns an unescaped string
    ----
    Paremeters:
      - text: string - the text string
    Returns:
      - the unescaped text string
    """
    return text.replace('\\_', '_').strip()
    

def __parse_label_index(label):
   """
   parses a label, checking if it contains a
   numbering suffix like LABEL[1].
   If so, it returns the label and the numbering index
   as a tuple
   ----
   Parameters:
      - label: string - the label
   Returns:
      the label and the index 
      If there is no index the index is -1
   """
   
   match = re.match(r'(.+)\[(\d+)\]', label)
   if match:
      try:
         label = match.group(1)
         index = int(match.group(2))
         return label, index
      except:
         pass
   return label, -1



    
def __position_labels(labels):
    """
    takes a sentence and converts labels according
    to the BIO-Syntax
    ----
    Parameters:
     - tags: list(str) - the tags of a sentence
    Returns:
     the converted tags list(str)
    """
    converted_labels = list()
    last_label_id = -1
    for label in labels:
        if label == 'O' or label == '_':
           converted_labels.append('O')
           continue
        raw_label, label_id = __parse_label_index(label)
        if label_id == -1 or label_id != last_label_id:
           converted_labels.append("B-" + raw_label)
        else: 
           converted_labels.append("I-" + raw_label)
        last_label_id = label_id
    return converted_labels
           

def __apply_window(*args, margin=1, offset=0):
   """
   apply a rolling window to a list of list (needed for
   sentence and tag lists) and return a new list of lists 
   with window-wise concatenated lists.
   usage: 
   d1, d2,... = __apply-window(list_of_lists_1, list_of_lists_2,...[, margin=2] [, offset=1])
   ---
   Parameters:
     - *args: an unlimited number of lists of lists
   returns:
     a list of dictionaries - each of the form:
     {'data': windowed_list, 'max_length': max_list_length}
   """
   if margin < 1 or offset < 0:
      raise Exception('Invalid margin or overlap')
   if offset > margin:
      raise Exception('offset must not be larger than margin')
   
   windowed_lists = list()
   for list_of_lists in args:
       max_list_length = 0   
       margined = list()
       list_length = len(list_of_lists)
       for i in range (0, list_length, offset + 1):
           start = max(0, i-margin)
           stop = min(list_length, i+margin+1)
           concatenated = list()
           for j in range(start, stop):
               concatenated += list_of_lists[j]
           margined.append(concatenated)
           max_list_length = max(max_list_length, len(concatenated))
       windowed_lists.append({'data': margined, 'max_length': max_list_length})
   return windowed_lists
   
   
def __flatten(*args):
    """
    Take lists of lists and for each make one big list out of it.
    usage: 
    l1, l2,... = __flatten(list_of_lists_1, list_of_lists_2,...)
    ---
    Parameters:
      - *args an unlimited number of lists of lists
    Returns:
      the flattened lists as tuple
    """
    
    concatenated_lists = list()
    for list_of_lists in args:
        concatenated = list()
        for sublist in list_of_lists:
            concatenated += sublist
        concatenated_lists.append(concatenated)
    return (*concatenated_lists, )
    
    
    
def webanno_to_ner_train_input(filepath, outfile=None, flatten=False, margin=0, offset=0):
    """
    Convert a webanno tsv file into a nerda-compatible
    JSON dictionary. The parameters margin and offset allow
    for setting a sliding window for aggregating multiple
    sentences.
    -----
    Parameters:
       filepath: string - the path to the tsv file
       outfile: string - path to the output json file (may be None)
       flatten: boolean - flatten all sentence lists to one single list of tokens and tags
       margin: int - determines how many sentences before and after
                     a sentence should be aggregated. This parameter is
                     not effective when flatten is set to True.
       offset: int - how many sentences to skip (useful if margin is set).
                     This parameter is not effective when flatten is set to True.
    Returns:
       a dictionary {'sentences': list(str), 'tags': list(str)}
    """
    nerdict = {'sentences': [], 'tags': [], 'metadata': {'max_sentence_length': 0}}
    
    df = __webanno_to_df(filepath)
    last_sentence_id = -1
    current_sentence = list()
    current_labels = list()

    for row in df.itertuples():
        sentence_id = __sentence_id(row.sentence_token)
        if not sentence_id: 
           continue
        if sentence_id != last_sentence_id:
           if last_sentence_id > -1:
              nerdict["sentences"].append(current_sentence)
              nerdict["tags"].append(__position_labels(current_labels))
           current_sentence = list()
           current_labels = list()
           last_sentence_id = sentence_id
        current_sentence.append(__unescape(row.token))
        current_labels.append(__unescape(row.label))
        nerdict["metadata"]["max_sentence_length"] = max(nerdict["metadata"]["max_sentence_length"], len(current_sentence))
        
    if not flatten and margin > 0:
       sentences, tags = __apply_window(nerdict["sentences"], nerdict["tags"], margin=margin, offset=offset)
       nerdict["sentences"] = sentences["data"]
       nerdict["tags"] = tags["data"]
       nerdict["metadata"]["max_sentence_length"] = sentences["max_length"]
    elif flatten:
       sentence, tags = __flatten(nerdict["sentences"], nerdict["tags"])
       nerdict["sentences"] = [sentence]
       nerdict["tags"] = [tags]
       nerdict["metadata"]["max_sentence_length"] = len(sentence)
    
    nerdict["metadata"]["infile"] = os.path.basename(filepath)
    nerdict["metadata"]["margin"] = margin
    nerdict["metadata"]["offset"] = offset
    nerdict["metadata"]["flatten"] = flatten
    
    if outfile:
       with open(outfile, 'w') as f:
            json.dump(nerdict, f, indent=5, ignore_nan=True)

    return nerdict
     
              
        
    
    
    

    


