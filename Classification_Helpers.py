import json
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tiktoken as tk
from openai import OpenAI
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

# Constants
MAX_TOKENS_GPT4 = 127000
MAX_TOKENS_GPT3 = 15600
RUNNING_COST = 0

MODEL = 'gpt-4-turbo'

# --------------------------------------------------------- #
# --------------- CFA 683 Specific Functions -------------- #
# --------------------------------------------------------- #

def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str,str]]:
    df = pd.read_excel(file_path, na_values=["NA", 'REF', ""])
    
    # Extracting the question number and the question from the column names
    qs = {x.split('.')[0]:''.join(x.split('. ')[1:]) for x in df.columns if x != 'NAID'}
    
    # Rename the columns to remove everything after the first period meaning we just keep the question number
    df.columns = df.columns.str.split('.').str[0]
    
    return df, qs

def normal_extract(client: OpenAI, df, col, n, context):
    context = f"Interviewees were asked to respond to the following prompt : '{context}'."
    
    normal_topics = extractCommonTopics(client, df[col].dropna().tolist(), n=n, model=MODEL, context=context)
    
    return normal_topics

def shuffle_extract(client: OpenAI, df: pd.DataFrame, col: str, n: int, context: str) -> List[str]:
    context = f"Interviewees were asked to respond to the following prompt : '{context}'."
    df = df.copy().sample(frac=1)
    
    shuffle_topics = extractCommonTopics(client, df[col].dropna().tolist(), n=n, model=MODEL, context=context)
    
    return sorted(shuffle_topics)   

def shuffle_extract_w_defs(client: OpenAI, df: pd.DataFrame, col: str, n: int, context: str, model=MODEL ) -> List[str]:
    context = f"Interviewees were asked to respond to the following prompt : '{context}'."
    df = df.copy().sample(frac=1)
    
    shuffle_topics = extractCommonTopicsWDefs(client, df[col].dropna().tolist(), n=n, model=model, context=context)
    # Sort the topics based on dictionary keys
    
    return {k: v for k, v in sorted(shuffle_topics.items(), key=lambda item: item[0])}   

def subset_extract(client: OpenAI, df: pd.DataFrame, col: str, n: int, context: str) -> List[str]:
    context = f"Interviewers provided a rating of if they agreed to the statement : '{context}'. They were then asked to provide their reasoning."
    
    shuffled_df = df.sample(frac=1)

    subset1 = shuffled_df.iloc[:int(len(shuffled_df)/3)]
    subset2 = shuffled_df.iloc[int(len(shuffled_df)/3):int(len(shuffled_df)/3)*2]
    subset3 = shuffled_df.iloc[int(len(shuffled_df)/3)*2:]
    
    subset1_topics = extractCommonTopics(client, subset1[col].dropna().tolist(), n=n, model=MODEL, context=context)
    subset2_topics = extractCommonTopics(client, subset2[col].dropna().tolist(), n=n, model=MODEL, context=context)
    subset3_topics = extractCommonTopics(client, subset3[col].dropna().tolist(), n=n, model=MODEL, context=context)
    
    return sorted(list(set(subset1_topics + subset2_topics + subset3_topics)))

def get_agree_disagree(df: pd.DataFrame, col: str, context: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    context = f"Interviewers provided a rating of if they agreed to the statement : '{context}'. They were then asked to provide their reasoning."
    rating_col = col[:-1]
    agree = df.dropna(subset=[col])[df.dropna(subset=[col])[rating_col].str[0].astype(int) > 3]
    disagree = df.dropna(subset=[col])[df.dropna(subset=[col])[rating_col].str[0].astype(int) <= 3]
    
    agree_context = f"{context}. The following responses are from interviewers who chose values 'Slighty agree', 'Agree', or 'Strongly Agree' for the rating."
    disagree_context = f"{context}. The following responses are from interviewers who chose values 'Slightly disagree', 'Disagree', or 'Strongly disagree' for the rating."

    return agree, disagree, agree_context, disagree_context


def agree_disagree_extract(client: OpenAI, df: pd.DataFrame, col: str, n_dict: Dict, context: str) -> List[str]:
    
    agree, disagree, agree_context, disagree_context = get_agree_disagree(df, col, context)
    
    agree_topics = extractCommonTopics(client, agree[col].dropna().tolist(), n=n_dict['Agree'], model=MODEL, context=agree_context)
    disagree_topics = extractCommonTopics(client, disagree[col].dropna().tolist(), n=n_dict['Disagree'], model=MODEL, context=disagree_context)
    
    return {'Agree': sorted(agree_topics), 'Disagree': sorted(disagree_topics)}


def paycheck_vs_passion_extract(client: OpenAI, df: pd.DataFrame, rating_col: str, a_col: str,
                                n_dict: Dict[str, int]) -> Dict[str, List[str]]:
    context = "Interviewers were asked if they worked for Chikfila due to Passion or Paycheck. They provided rankings from 1-10."
    df2 = df.copy().dropna(subset=[a_col])
    
    paycheck = df2[df2[rating_col].str[:2].astype(int) <= 5]
    passion = df2[df2[rating_col].str[:2].astype(int) >= 9]
    neutral = df2[(df2[rating_col].str[:2].astype(int) > 5) & (df2[rating_col].str[:2].astype(int) < 9)]
    
        
    paycheck_context = f"{context}. The following responses are from interviewers who chose rankings corresponding to 'Paycheck' for their rating."
    passion_context = f"{context}. The following responses are from interviewers who chose rankings corresponding to 'Passion' for their rating."
    neutral_context = f"{context}. The following responses are from interviewers who chose rankings corresponding to 'Neutral' for their rating."
    
    paycheck_topics = extractCommonTopics(client, paycheck[a_col].tolist(), n=n_dict['Paycheck'], model=MODEL, context=paycheck_context)
    passion_topics = extractCommonTopics(client, passion[a_col].tolist(), n=n_dict['Passion'], model=MODEL, context=passion_context)
    neutral_topics = extractCommonTopics(client, neutral[a_col].tolist(), n=n_dict['Neutral'], model=MODEL, context=neutral_context)
    
    return {'Paycheck': sorted(paycheck_topics), 'Passion': sorted(passion_topics), 'Neutral': sorted(neutral_topics)}


# --------------------------------------------------------- #
# -------------- Data Manipulation Functions -------------- #
# --------------------------------------------------------- #

def turnClassesLong(df: pd.DataFrame, ids: List[str], sort_col: List[str]=['ID', 'Class_No']) -> pd.DataFrame:
    """Turn a wide dataframe of classes into a long dataframe of classes.

    Args:
        df (pd.DataFrame): Input dataframe with classes in wide format.
        ids (List[str]): List of columns to keep as ids.
        sort_col (List[str], optional): List of columns to sort by. Defaults to ['NAID', 'Class_No'].

    Returns:
        pd.DataFrame: Long dataframe of classes.
    """

    # Get Class columns
    class_cols = df.filter(regex='Class').columns
    df = df.melt(id_vars=ids, 
                 value_vars=class_cols,
                 value_name = "Label",
                 var_name = "Class_No").sort_values(sort_col)

    # Remove all NaN in Label No
    df = df[df['Label'].notna()]
    
    return df

def wide_encode_output(df: pd.DataFrame, id_cols: List[str], bin_rank: str = 'binary'):
    if bin_rank not in ['binary', 'rank']:
        raise ValueError("bin_rank must be one of 'binary' or 'rank'")
    
    class_vars = df.filter(regex='Class_').columns.tolist()
    melted_df = df.melt(id_vars= id_cols, value_vars=class_vars, value_name='Class').dropna()
    melted_df['variable'] = melted_df['variable'].str.extract('(\d+)')
    
    if bin_rank == 'binary':
        melted_df['variable'] = 1
    
    pivoted_df = melted_df.pivot(index= id_cols, columns='Class', values='variable').fillna(0).reset_index()
    
    return pivoted_df.apply(lambda col: col.astype(int) if col.name not in id_cols else col)

# --------------------------------------------------------- #
# -------------- GPT Specific Helper Functions ------------- #
# --------------------------------------------------------- #

def logAttemptNumber(retry_state: RetryCallState):
    """Log the attempt number and outcome of a retry operation.
    
       Args:
        retry_state (RetryCallState): The state of the retry operation.

    Returns:
        None
    """
    # logging.error(f"Retrying {retry_state.attempt_number} due to Error: {retry_state.outcome}")
    print(f"ERROR: Retrying {retry_state.attempt_number} due to Error: {retry_state.outcome}")
    
def getNumTokens(txt: str, model: str='gpt-3.5-turbo') -> int:
    """
    Returns the number of tokens in a text string for a given model encoding.

    Args:
        txt (str): The input text.
        model (str): The name of the model for encoding.

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tk.encoding_for_model(model)
    return len(encoding.encode(txt))

@retry(wait=wait_fixed(30), stop=stop_after_attempt(2),after=logAttemptNumber)
def extractCommonTopics(client: OpenAI, verbatims: pd.Series, n:int, model: str, context: str) -> str:
    """
    Extract the {n} most common topics from a large set of text verbatims using an OpenAI Chat Model.

    Args:
        verbatims (pd.Series): A Pandas Series of verbatims. Could also be a list. 
        n (int): The number of topics to extract.
        model (str): The name of the language model to use. Must be one of the OpenAI Chat Models
        context (str): Context about the topic to be given to the model about what the verbatims are discussing.

    Returns:
        str: The extracted key topics as a Python array of strings.
    """
    global RUNNING_COST
    # Combine verbatims
    verbatims = r"/n-/n".join(verbatims) 
    
    # Set up the prompt
    #                 The following topics have already been extracted from the responses and should be included in the final list of topics and not duplicated:
                # •	Connection with Others
                # •	Balancing Productivity
                # •	Effective Communication
                # •	Work-Life Balance
                # •	Designated Office Days
                # •	Technology Concerns
    prompt = f"""
                {context} From the following open-ended interview responses, extract the {n} most common topics discussed. Only include the {n} most common topics and label the topics with 4 or less words.
                
                Return a JSON object of the following form: """ + \
                """ 
                    {'topics': ["Topic 1", "Topic 2", "Topic 3", ...] }
                    
                Each response below is separated by "/n-/n". /n 
                
                Responses: |
                """ + verbatims + " \n |"
    # Check number of tokens in prompt  
    # We need to make sure the model has a large enough context window for all the verbatims
    n_tokens = getNumTokens(prompt)                
    print(f"Number of Tokens in Prompt: {n_tokens}") 
    
    if  n_tokens > MAX_TOKENS_GPT4:
        raise ValueError(f"Too many tokens for any model. Clean up the verbatims or chunk them. Max tokens is {MAX_TOKENS_GPT4}")
    
            
    # Ask the large language model to extract the topics
    gpt_resp =  client.chat.completions.create(
        model = model,
        messages = [{"role": "system", "content": "You are an expert in topic modeling and extracting key topics from open ended survey responses."},
                    {"role": "user", "content": prompt}],
        temperature = 0,
        top_p = 0.1,
        n = 1, 
        stop = None,
        response_format={'type': 'json_object'}
    )
    
    cost = (gpt_resp.usage.completion_tokens * 0.03 / 1000) + (gpt_resp.usage.prompt_tokens * 0.01 / 1000)
    RUNNING_COST += cost
    
    print(f'Cost For Running Topic Extraction: ${cost:.2f}')

    topics = json.loads(gpt_resp.choices[0].message.content)

    return topics['topics']


@retry(wait=wait_fixed(30), stop=stop_after_attempt(2),after=logAttemptNumber)
def extractCommonTopicsWDefs(client: OpenAI, verbatims: pd.Series, n:int, model: str, context: str) -> Dict[str, str]:
    """
    Extract the {n} most common topics from a large set of text verbatims using an OpenAI Chat Model.

    Args:
        verbatims (pd.Series): A Pandas Series of verbatims. Could also be a list. 
        n (int): The number of topics to extract.
        model (str): The name of the language model to use. Must be one of the OpenAI Chat Models
        context (str): Context about the topic to be given to the model about what the verbatims are discussing.

    Returns:
        str: The extracted key topics as a Python array of strings.
    """
    global RUNNING_COST
    # Combine verbatims
    verbatims = r"/n-/n".join(verbatims) 
    
    # Set up the prompt
    # {'topics': ["Topic 1", "Topic 2", "Topic 3", ...] }
    prompt = f"""
                {context} From the following open-ended interview responses, extract the {n} most common topics discussed. Only include the {n} most common topics and label the topics with 4 or less words.
                
                For each extracted topic, include a 1-2 description of what types of responses the given topic encompasses. 

                Return a JSON object of the following form: """ + \
                """ 
                    {
                        "topics":
                            {"Topic 1": "Description of Topic 1",
                             "Topic 2": "Description of Topic 2",
                             "Topic 3": "Description of Topic 3",
                             ...
                            }
                    }
                    
                Each response below is separated by "/n-/n". /n 
                
                Responses: |
                """ + verbatims + " \n |"
    # Check number of tokens in prompt  
    # We need to make sure the model has a large enough context window for all the verbatims
    n_tokens = getNumTokens(prompt)                
    print(f"Number of Tokens in Prompt: {n_tokens}") 
    
    if  n_tokens > MAX_TOKENS_GPT4:
        raise ValueError(f"Too many tokens for any model. Clean up the verbatims or chunk them. Max tokens is {MAX_TOKENS_GPT4}")
    
            
    # Ask the large language model to extract the topics
    gpt_resp =  client.chat.completions.create(
        model = model,
        messages = [{"role": "system", "content": "You are an expert in topic modeling and extracting key topics from open ended survey responses."},
                    {"role": "user", "content": prompt}],
        temperature = 0,
        top_p = 0.1,
        n = 1, 
        stop = None,
        response_format={'type': 'json_object'}
    )
    
    cost = (gpt_resp.usage.completion_tokens * 0.03 / 1000) + (gpt_resp.usage.prompt_tokens * 0.01 / 1000)
    RUNNING_COST += cost
    
    print(f'Cost For Running Topic Extraction: ${cost:.2f}')

    topics = json.loads(gpt_resp.choices[0].message.content)

    return topics['topics']

def createMultiLabelPrompt(text_input: str, labels: List[str], context: str, output_format: str =None, n_max: int = 5) -> tuple[str, str]:
    """Create the prompt for multilabel classification

    Args:
        text_input (str): Input verbatim to classify
        labels (List[str]): List of classification topics for the model to choose from
        context (str): Context about the topic to be given to the model about what the verbatims are discussing.
        output_format (str, optional): Format for GPT To output
        examples (List[str], optional): One or multi-shot examples to provide the model
        n_max (int, optional): Maximum number of topics for the model to classify a single verbatim. Defaults to 5.

    Returns:
        tuple[str, str]: both the role of the model and the prompt
    """
    
    # Setting the system role
    role = f"""You are a highly intelligent and accurate multi-label classification system. You take a passage as input and classify it into at most {n_max} appropriate classes from a given category list. Only use topics that are truly appropriate, even if that means you only classify a response with one topic. 
                If none of the topics apply to the response, return 'Other' as the main class and do not include further categories. Once again, only select topics that truly align with the response. 
    """
    
    # Setting the main description
    prompt = f"""Below is from an excerpt from an interview. {context}.
                You are to classify the response into at most {n_max} appropriate categories. If none of the topics apply to the response, return 'Other' as the main class and do not include further categories. Once again, only select topics that without a doubt align with the response. 
                
                Here are the topics you can choose from: {str(labels)}
                 
    """

    output_format = output_format or "{'main_class': Main Classification Category ,'Class_2': 2nd level Classification Category, 'Class_3': 3rd level Classification Category, ...}"
    prompt += f"Your output format is a JSON object of the form {output_format}.\n"

    prompt += f"Input: {text_input}\nOutput:"

    return role, prompt

@retry(wait=wait_fixed(30), stop=stop_after_attempt(4),after=logAttemptNumber)
def multiclassifyVerbatim(client: OpenAI, verbatim: str, topics: List[str], context: str, id: int, model: str = MODEL, i:int = None, max_labels = 4, n_resp = 1) -> pd.DataFrame:
    """Classify a verbatim into one or more of the given topics using a language model.

    Args:
        verbatim (str): The verbatim text to classify.
        topics (List[str]): List of available topics.
        context (str): Context about the topic to be given to the model about what the verbatims are discussing.
        model (str): The name of the language model to use.
        id (int): The id of the verbatim
        i (int, optional): Verbatim number for printing progress
        max_labels (int, optional): Maximum number of topics for the model to classify a single verbatim. Defaults to 4.
        n_resp (int, optional): Number of Resposnes to Generate. Used for consistency analysis

    Returns:
        pd.DataFrame: Wide dataframe of classifications
    """
    global RUNNING_COST
    if i is not None:
        print(f'Classifying Verbatim Number: {i + 1}')
        
    # Set up the prompt and role  
    role, prompt = createMultiLabelPrompt(labels = topics, 
                                          text_input=verbatim,
                                          context=context,
                                          n_max = max_labels)   
    
    gpt_resp =  client.chat.completions.create(
        model = model,
        messages = [{"role": "system", "content": role},
                    {"role": "user", "content": prompt}],
        temperature = 0,
        top_p = 0.1,
        n = n_resp, 
        stop = None,
        response_format={'type': 'json_object'}
    )
    
    cost = (gpt_resp.usage.completion_tokens * 0.03 / 1000) + (gpt_resp.usage.prompt_tokens * 0.01 / 1000)
    RUNNING_COST += cost
    # resp = resp.replace("'s", "s") # Remove apostrophe s from the model output because it messes up the eval function
    labels = json.loads(gpt_resp.choices[0].message.content)
    
    # Make sure all labels are in the labels list
    attempts = 1
    not_in = [label for label in labels.values() if label not in topics + ['Other']]
    
    
    while len(not_in) > 0 and attempts < 4:
        print(not_in)
        print(topics)
        
        gpt_resp =  client.chat.completions.create(
        model = model,
        messages = [{"role": "system", "content": role},
                    {"role": "user", "content": prompt}],
        temperature = 0.5, # Increase temperature to get more diverse responses
        top_p = 0.5,
        n = n_resp, 
        stop = None,
        response_format={'type': 'json_object'}
    )
    
        cost = (gpt_resp.usage.completion_tokens * 0.03 / 1000) + (gpt_resp.usage.prompt_tokens * 0.01 / 1000)
        RUNNING_COST += cost
        # resp = resp.replace("'s", "s") # Remove apostrophe s from the model output because it messes up the eval function
        labels = json.loads(gpt_resp.choices[0].message.content)
        not_in = [label for label in labels.values() if label not in topics]
        attempts += 1
    
    if attempts == 3:
        print(f"Could not get the right labels for verbatim {id}")
        # Replace the bad labels with 'Other'
        for key in labels.keys():
            if labels[key] not in topics:
                if 'Other' in labels.values():
                    labels[key] = np.nan
                else:
                    labels[key] = 'Other'
    elif attempts > 1:
        print(f"Got the right labels for verbatim {id} on attempt {attempts}")
    
    # Convert the output to a dataframe
    df: pd.DataFrame = pd.DataFrame([id], columns=['NAID'])
    for key in labels.keys():
        if key == 'main_class':
            df['Class_1'] = labels[key]
        else:
            df[key] = labels[key]
            
    return df
    
    
def get_running_cost():
    global RUNNING_COST
    return RUNNING_COST