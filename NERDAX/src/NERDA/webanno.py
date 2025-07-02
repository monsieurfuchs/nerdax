import pandas as pd
import re


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
    
    
    
def __parse_label(label):
    """
    parses a label entry and returns an unescaped label string
    ----
    Paremeters:
      - label: string - the label string
    Returns:
      - the unescaped label as string ('O' if there is no label)
    """
    return label.replace('\\_', '_').strip()
    

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
           

def __apply_window(list_of_lists, margin=1, offset=0):
   """
   apply a rolling window to a list of list (needed for
   sentence and tag lists) and return a new list of lists 
   with window-wise concatenated lists.
   ---
   Parameters:
     - list_of_lists: list - may be a list of sentences or tags
   returns:
     - the new list of lists
   """
   if margin < 1 or offset < 0:
      raise Exception('Invalid margin or overlap')
   if offset > margin:
      raise Exception('offset must not be larger than margin')
   
   margined = list()
   max_length = len(list_of_lists)
   for i in range (0, max_length, offset + 1):
       start = max(0, i-margin)
       stop = min(max_length, i+margin+1)
       concatenated = list()
       for j in range(start, stop):
           concatenated += list_of_lists[j]
       margined.append(concatenated)
   return margined
    
    
def webanno_to_ner_train_input(filepath, margin=0, offset=0):
    """
    Convert a webanno tsv file into a nerda-compatible
    JSON dictionary.
    -----
    Parameters:
       filepath: string - the path to the tsv file
    Returns:
       a dictionary {'sentences': list(str), 'tags': list(str)}
    """
    nerdict = dict()
    nerdict['sentences'] = list()
    nerdict['tags'] = list()
    
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
        current_sentence.append(row.token)
        current_labels.append(__parse_label(row.label))
        
    if margin > 0:
       nerdict["sentences"] = __apply_window(nerdict["sentences"], margin=margin, offset=offset)
       nerdict["tags"] = __apply_window(nerdict["tags"], margin=margin, offset=offset)
        
    return nerdict
     
              
        
    
    
    

    


