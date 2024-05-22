# -*- coding: utf-8 -*-
"""
### STRING BASED ANNOTATION - inclusion of seed per iteration

## Overview
Given a codebook (.txt) and a dataset (.csv) that has one text column and any number of category columns as binary indicators, the main function (`gpt_annotate`) annotates
all the samples using an OpenAI GPT model (ChatGPT or GPT-4) and calculates performance metrics (if there are provided human labels). Before running `gpt_annotate`,
users should run `prepare_data` to ensure that their data is in the correct format.

Flow of `gpt_annotate`:
*   1) Based on a provided codebook, the function uses an OpenAI GPT model to annotate every text sample per iteration, which is a parameter set by user.
*   2) The function reduces the annotation output down to the modal annotation category across iterations for each category. At this stage,
       the function adds a consistency score for each annotation across iterations.
*   3) If provided human labels, the function determines, for every category, whether the annotation is correct (by comparing to the human label),
        then also adds whether it is a true positive, false positive, true negative, or false negative.
*   4) Finally, if provided human labels, the function calculates performance metrics (accuracy, precision, recall, and f1) for every category.

The main function (`gpt_annotate`) returns one .csv
*   1) `gpt_out_all_iterations.csv`
  *   Raw outputs for every iteration.

Our code aims to streamline automated text annotation for different datasets and numbers of categories.

"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("openai")
install("pandas")
install("numpy")
install("tiktoken")

import openai
import pandas as pd
import math
import time
import numpy as np
import tiktoken
from openai import OpenAI
import os

# Create a global fingerprint dataframe
fingerprints = pd.DataFrame()

def prepare_data(text_to_annotate, codebook, key,
                 prep_codebook = False, human_labels = True, no_print_preview = False):

  """
  This function ensures that the data is in the correct format for LLM annotation. 
  If the data fails any of the requirements, returns the original input dataframe.

  text_to_annotate: 
      Data that will be prepared for analysis. Should include a column with text to annotate, and, if human_labels = True, the human labels.
  codebook: 
      String detailing the task-specific instructions.
  key:
    OpenAI API key
  prep_codebook: 
      boolean indicating whether to standardize beginning and end of codebook to ensure that the LLM prompt is annotating text samples.
  human_labels: 
      boolean indicating whether text_to_annotate has human labels to compare LLM outputs to. 
  no_print_preview:
    Does not print preview of user's data after preparing.

  Returns:
    Updated dataframe (text_to_annotate) and codebook (if prep_codebook = True) that are ready to be used for annotation using gpt_annotate.
  """

  # Check if text_to_annotate is a dataframe
  if not isinstance(text_to_annotate, pd.DataFrame):
    print("Error: text_to_annotate must be pd.DataFrame.")
    return text_to_annotate

  # Make copy of input data
  original_df = text_to_annotate.copy()

  # set OpenAI key
  openai.api_key = key

  # Standardize beginning and end of codebook to ensure that the LLM prompt is annotating text samples 
  if prep_codebook == True:
    codebook = prepare_codebook(codebook)

  # Add unique_id column to text_to_annotate
  text_to_annotate = text_to_annotate \
                    .reset_index() \
                    .rename(columns={'index':'unique_id'})

  ##### Minor Cleaning
  # Drop any Unnamed columns
  if any('Unnamed' in col for col in text_to_annotate.columns):
    text_to_annotate = text_to_annotate.drop(text_to_annotate.filter(like='Unnamed').columns, axis=1)
  # Drop any NA values
  text_to_annotate = text_to_annotate.dropna()

  ########## Confirming data is in correct format
  ##### 1) Check whether second column is string
  # rename second column to be 'text'
  text_to_annotate.columns.values[1] = 'text'
  # check whether second column is string
  if not (text_to_annotate.iloc[:, 1].dtype == 'string' or text_to_annotate.iloc[:, 1].dtype == 'object'):
    print("ERROR: Second column should be the text that you want to annotate.")
    print("")
    print("Your data:")
    print(text_to_annotate.head())
    print("")
    print("Sample data format:")
    error_message(human_labels)
    return original_df
  
  ##### 2) If human_labels == False, there should only be 2 columns
  if human_labels == False and len(text_to_annotate.columns) != 2:
    print("ERROR: You have set human_labels = False, which means you should only have two columns in your data.")
    print("")
    print("Your data:")
    print(text_to_annotate.head())
    print("")
    print("Sample data format:")
    error_message(human_labels)
    return original_df

  ##### 3) If human_labels == True, there should be more than 2 columns
  if human_labels == True and len(text_to_annotate.columns) < 3:
    print("ERROR: You have set human_labels = True (default value), which means you should have more than 2 columns in your data.")
    print("")
    print("Your data:")
    print(text_to_annotate.head())
    print("")
    print("Sample data format:")
    error_message(human_labels)
    return original_df

  ##### 5) Add llm_query column that includes a unique ID identifier per text sample
  text_to_annotate['llm_query'] = text_to_annotate.apply(lambda x: str(x['unique_id']) + " " + str(x['text']) + "\n", axis=1)

  ##### 6) Make sure category names in codebook exactly match category names in text_to_annotate
  # extract category names from codebook
  if human_labels:
    col_names_codebook = get_classification_categories(codebook, key)
    # get category names in text_to_annotate
    df_cols = text_to_annotate.columns.values.tolist()
    # remove 'unique_id', 'text' and 'llm_query' from columns
    col_names = [col for col in df_cols if col not in ['unique_id','text', 'llm_query']]

    ### Check whether categories are the same in codebook and text_to_annotate
    if [col for col in col_names] != col_names_codebook:
      print("ERROR: Column names in codebook and text_to_annotate do not match exactly. Please note that order and capitalization matters.")
      print("Change order/spelling in codebook or text_to_annotate.")
      print("")
      print("Exact order and spelling of category names in text_to_annotate: ", col_names)
      print("Exact order and spelling of category names in codebook: ", col_names_codebook)
      return original_df
  else:
    col_names = get_classification_categories(codebook, key)

  ##### Confirm correct categories with user
  # Print annotation categories
  print("")
  print("Categories to annotate:")
  for index, item in enumerate(col_names, start=1):
    print(f"{index}) {item}")
  print("")
  if no_print_preview == False:
    waiting_response = True
    while waiting_response:
      # Confirm annotation categories
      input_response = input("Above are the categories you are annotating. Is this correct? (Options: Y or N) ")
      input_response = str(input_response).lower()
      if input_response == "y" or input_response == "yes":
        print("")
        print("Data is ready to be annotated using gpt_annotate()!")
        print("")
        print("Glimpse of your data:")
        # print preview of data
        print("Shape of data: ", text_to_annotate.shape)
        print(text_to_annotate.head())
        return text_to_annotate
      elif input_response == "n" or input_response == "no":
        print("")
        print("Adjust your codebook to clearly indicate the names of the categories you would like to annotate.")
        return original_df
      else:
        print("Please input Y or N.")
  else:
    return text_to_annotate


def gpt_annotate(text_to_annotate, codebook, key, seed,
                 num_iterations = 3, model = "gpt-4", temperature = 0.6, batch_size = 10,
                 human_labels = True,  data_prep_warning = True,
                 time_cost_warning = True):
  """
  Loop over the text_to_annotate rows in batches and classify each text sample in each batch for multiple iterations. 
  Store outputs in a csv. Function is calculated in batches in case of crash.

  text_to_annotate:
    Input data that will be annotated.
  codebook:
    String detailing the task-specific instructions.
  key:
    OpenAI API key.
  num_iterations:
    Number of times to classify each text sample.
  model:
    OpenAI GPT model, which is either gpt-3.5-turbo or gpt-4
    base model is set to 3.5-turbo - 16-5
  temperature: 
    LLM temperature parameter (ranges 0 to 1), which indicates the degree of diversity to introduce into the model.
  batch_size:
    number of text samples to be annotated in each batch.
  human_labels: 
    boolean indicating whether text_to_annotate has human labels to compare LLM outputs to. 
  data_prep_warning: 
    boolean indicating whether to print data_prep_warning
  time_cost_warning: 
    boolean indicating whether to print time_cost_warning

  Returns:
    gpt_annotate returns the four .csv's below, if human labels are provided. If no human labels are provided, 
    gpt_annotate only returns gpt_out_all_iterations.csv and gpt_out_final.csv
    
    1) `gpt_out_all_iterations.csv`
        Raw outputs for every iteration.
    2) `gpt_out_final.csv`
        Annotation outputs after taking modal category answer and calculating consistency scores.
    3) `performance_metrics.csv`
        Accuracy, precision, recall, and f1.
    4) `incorrect.csv`
        Any incorrect classification or classification with less than 1.0 consistency.

  """

  from openai import OpenAI

  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)

  # set OpenAI key
  openai.api_key = key

  # Double check that user has confirmed format of the data
  if data_prep_warning:
    waiting_response = True
    while waiting_response:
      input_response = input("Have you successfully run prepare_data() to ensure text_to_annotate is in correct format? (Options: Y or N) ")
      input_response = str(input_response).lower()
      if input_response == "y" or input_response == "yes":
        # If user has run prepare_data(), confirm that data is in correct format
        # first, check if first column is title "unique_id"
        if text_to_annotate.columns[0] == 'unique_id' and text_to_annotate.columns[-1] == 'llm_query':
            try:
              text_to_annotate_copy = text_to_annotate.iloc[:, 1:-1]
            except pd.core.indexing.IndexingError:
              print("")
              print("ERROR: Run prepare_data(text_to_annotate, codebook, key) before running gpt_annotate(text_to_annotate, codebook, key).")
            text_to_annotate_copy = prepare_data(text_to_annotate_copy, codebook, key, prep_codebook = False, human_labels = human_labels, no_print_preview = True)
            # if there was an error, exit gpt_annotate
            if text_to_annotate_copy.columns[0] != "unique_id":
              if human_labels:
                return None, None, None, None
              elif human_labels == False:
                return None, None
        else:
            print("ERROR: First column should be title 'unique_id' and last column should be titled 'llm_query'")
            print("Try running prepare_data() again")
            print("")
            print("Your data:")
            print(text_to_annotate.head())
            print("")
            print("Sample data format:")
            if human_labels:
                return None, None, None, None
            elif human_labels == False:
                return None, None
        waiting_response = False
      elif input_response == "n" or input_response == "no":
        print("")
        print("Run prepare_data(text_to_annotate, codebook, key) before running gpt_annotate(text_to_annotate, codebook, key).")
        if human_labels:
            return None, None, None, None
        elif human_labels == False:
            return None, None
      else:
        print("Please input Y or N.")

  # df to store results
  out = pd.DataFrame()

  # Determine number of batches
  num_rows = len(text_to_annotate)
  # Round upwards to ensure that all rows are included.
  num_batches = math.ceil(num_rows/batch_size)
  num_iterations = num_iterations

  # Add categories to classify
  col_names = ["unique_id"] + text_to_annotate.columns.values.tolist()[2:-1]
  if human_labels == False:
    col_names = get_classification_categories(codebook, key)
    col_names = ["unique_id"] + col_names

  ### Nested for loop for main function
  # Iterate over number of classification iterations
  for j in range(num_iterations):
    # Iterate over number of batches
    for i in range(num_batches):
      # Based on batch, determine starting row and end row
      start_row = i*batch_size
      end_row = (i+1)*batch_size

      # Extract the text samples to annotate
      llm_query = text_to_annotate['llm_query'][start_row:end_row].str.cat(sep=' ')

      # Start while loop in case GPT fails to annotate a batch
      need_response = True
      while need_response:
        fails = 0
        # confirm time and cost with user before annotating data
        if fails == 0 and j == 0 and i == 0 and time_cost_warning:
          quit = estimate_time_cost(text_to_annotate, codebook, llm_query, model, num_iterations, num_batches, batch_size, col_names[1:])
          if quit and human_labels:
            return None, None, None, None
          elif quit and human_labels == False:
            return None, None
        # if GPT fails to annotate a batch 3 times, skip the batch
        while(fails < 3):
          try:
            # Set temperature
            temperature = temperature
            # set seed
            seed = seed
            # annotate the data by prompting GPT
            response = get_response(codebook, llm_query, model, temperature, seed, key)
            # parse GPT's response into a clean dataframe
            text_df_out = parse_text(response, col_names)
            break
          except:
            fails += 1
            pass
        if (',' in response.choices[0].message.content  or '|' in response.choices[0].message.content):
          need_response = False 

      # update iteration
      text_df_out['iteration'] = j+1

      # add iteration annotation results to output df
      out = pd.concat([out, text_df_out])
      time.sleep(.5)
    # print status report  
    print("iteration: ", j+1, "completed")

  # Save fingerprints dataframe to CSV
  global fingerprints
  fingerprints.to_csv('fingerprints_mainseed')

  # Convert unique_id col to numeric - account for non-numeric values
  out['unique_id'] = pd.to_numeric(out['unique_id'], errors='coerce')

  # Combine input df (i.e., df with text column and true category labels)
  out_all = pd.merge(text_to_annotate, out, how="inner", on="unique_id")

  # replace any NA values with 0's
  out_all.replace('', np.nan, inplace=True)
  out_all.replace('-', np.nan, inplace=True)
  out_all.fillna(0, inplace=True)

  ##### output 1: full annotation results
  out_all.to_csv('gpt_out_all_iterations_string.csv',index=False)

  ## There is no evaluation agains true labels
  return out_all



########### Helper Functions

def prepare_codebook(codebook):
  """
  Standardize beginning and end of codebook to ensure that the LLM prompt is annotating text samples. 

  codebook: 
      String detailing the task-specific instructions.

  Returns:
    Updated codebook ready for annotation.
  """
  beginning = "Use this codebook for text classification. Return your classifications in a table with one column for text number (the number preceding each text sample) and a column for each label. Use a csv format. "
  end = " Classify the following text samples:"
  return beginning + codebook + end

def error_message(human_labels = True):
  """
  Prints sample data format if error.
  
  human_labels: 
      boolean indicating whether text_to_annotate has human labels to compare LLM outputs to.
  """
  if human_labels == True:
    toy_data = {
      'unique_id': [0, 1, 2, 3, 4],
      'text': ['sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate'],
      'category_1': [1, 0, 1, 0, 1],
      'category_2': [0, 1, 1, 0, 1]
      }
    toy_data = pd.DataFrame(toy_data)
    print(toy_data)
  else:
    toy_data = {
        'unique_id': [0, 1, 2, 3, 4],
      'text': ['sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate', 'sample text to annotate'],
      }
    toy_data = pd.DataFrame(toy_data)
    print(toy_data)

def get_response(codebook, llm_query, model, temperature, seed, key):
  """
  Added seed element - function is used when annotating sentences

  Function to query OpenAI's API and get an LLM response.

  Codebook: 
    String detailing the task-specific instructions
  llm_query: 
    The text samples to append to the task-specific instructions
  Model: 
    gpt-3.5-turbo (Chat-GPT) or GPT-4
  Temperature: 
    LLM temperature parameter (ranges 0 to 1)

  Returns:
    LLM output.
  """

  from openai import OpenAI
  
  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)
  
  # Set max tokens, to be the same for every response
  max_tokens = 4000

  # Create function to llm_query GPT - all parameters are the same for each batch
  response = client.chat.completions.create(
    model=model, # chatgpt: gpt-3.5-turbo # gpt-4: gpt-4
    messages=[
      {"role": "user", "content": codebook + llm_query}],
    seed = seed, # add seed parameter, followed per iteration (I hope)
    temperature=temperature, # ChatGPT default is 0.7 (set lower reduce variation across queries)
    max_tokens = max_tokens,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )

  # Save fingerprint of each analysis
  system_fingerprint = response.system_fingerprint
  global fingerprints  # Access the global DataFrame
  new_row = pd.DataFrame({"System_Fingerprint": [system_fingerprint]})
  fingerprints = pd.concat([fingerprints, new_row], ignore_index=True)

  return response

def get_classification_categories(codebook, key):
  """
  Function that extracts what GPT will label each annotation category to ensure a match with text_to_annotate.
  Order and exact spelling matter. Main function will not work if these do not match perfectly.

  Codebook: 
    String detailing the task-specific instructions

  Returns:
    Categories to be annotated, as specified in the codebook.
  """

  # llm_query to ask GPT for categories from codebook
  llm_query = "Part 2: I've provided a codebook in the previous sentences. Please print the category names in the order you will classify them. Ignore every other task that I described in the codebook.   I only want to know the categories. Do not include any text numbers or any annotations in your response. Do not include any language like 'The categories to be identified are:'. Only include the names of the categories you are identifying. : "

  # Set temperature to 0 to make model deterministic
  temperature = 0

  # Specify model to use
  #model = "gpt-3.5-turbo-0125"
  model = "gpt-4o"

  #As of may 7th seems to be the only gpt that returns a fingerprint
  #model= "gpt-4-turbo-2024-04-09"

  # Set seed for category determination as 1
  seed = 1

  from openai import OpenAI
  
  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)

  ### Get GPT response and clean response
  response = get_response(codebook, llm_query, model, temperature, seed, key)

  # print full response
  print(response)

  text = response.choices[0].message.content
  text_split = text.split('\n')
  text_out = text_split[0]
  # text_out_list is final output of categories as a list
  codebook_columns = text_out.split(', ')

  return codebook_columns

def parse_text(response, headers):
  """
  This function converts GPT's output to a dataframe. GPT sometimes returns the output in different formats.
  Because there is variability in GPT outputs, this function handles different variations in possible outputs.

  response:
    LLM output
  headers:
    column names for text_to_annotate dataframe

  Returns:
    GPT output as a cleaned dataframe.

  """
  try:
    text = response.choices[0].message.content
    text_split = text.split('\n')

    if any(':' in element for element in text_split):
      text_split_split = [item.split(":") for item in text_split]

    if ',' in text:
      text_split_out = [row for row in text_split if (',') in row]
      text_split_split = [text.split(',') for text in text_split_out]
    if '|' in text:
      text_split_out = [row for row in text_split if ('|') in row]
      text_split_split = [text.split('|') for text in text_split_out]

    for row in text_split_split:
      if '' in row:
        row.remove('')
      if '' in row:
        row.remove('')
      if ' ' in row:
        row.remove(' ')

    text_df = pd.DataFrame(text_split_split)
    # Try for inclusion of all columns
    text_df_out = pd.DataFrame(text_df.values, columns=headers)
    text_df_out = text_df_out[text_df_out.iloc[:,1].astype(str).notnull()]

    # Remove checking for numerical values
    #text_df_out = text_df_out[pd.to_numeric(text_df_out.iloc[:,1], errors='coerce').notnull()]

  except Exception as e:
    print(
      "ERROR: GPT output not in specified categories. Make your codebook clearer to indicate what the output format should be.")
    print("Try running prepare_data(text_to_annotate, codebook, key, prep_codebook = True")
    print("")
  return text_df_out

def num_tokens_from_string(string: str, encoding_name: str):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def estimate_time_cost(text_to_annotate, codebook, llm_query, 
                       model, num_iterations, num_batches, batch_size, col_names):
  """
  This function estimates the cost and time to run gpt_annotate().

  text_to_annotate:
    Input data that will be annotated.
  codebook:
    String detailing the task-specific instructions.
  llm_query:
    Codebook plus the text samples in batch that will annotated.
  model:
    OpenAI GPT model, which is either gpt-3.5-turbo or gpt-4
  num_iterations:
    Number of iterations in gpt_annotate
  num_batches:
    number of batches in gpt_annotate
  batch_size:
    number of text samples in each batch
  col_names:
    Category names to be annotated

  Returns:
    quit, which is a boolean indicating whether to continue with the annotation process.
  """
  # input estimate
  num_input_tokens = num_tokens_from_string(codebook + llm_query, "cl100k_base")
  total_input_tokens = num_input_tokens * num_iterations * num_batches
  if model == "gpt-4":
    gpt4_prompt_cost = 0.00003
    prompt_cost = gpt4_prompt_cost * total_input_tokens
  else:
    chatgpt_prompt_cost = 0.000002
    prompt_cost = chatgpt_prompt_cost * total_input_tokens

  # output estimate
  num_categories = len(text_to_annotate.columns)-3 # minus 3 to account for unique_id, text, and llm_query
  estimated_output_tokens = 3 + (5 * num_categories) + (3 * batch_size * num_categories) # these estimates are based on token outputs from llm queries
  total_output_tokens = estimated_output_tokens * num_iterations * num_batches
  if model == "gpt-4":
    gpt4_out_cost = 0.00006
    output_cost = gpt4_out_cost * total_output_tokens
  else:
    chatgpt_out_cost = 0.000002
    output_cost = chatgpt_out_cost * total_output_tokens

  cost = prompt_cost + output_cost
  cost_low = round(cost*0.9,2)
  cost_high = round(cost*1.1,2)

  if model == "gpt-4":
    time = round(((total_input_tokens + total_output_tokens) * 0.02)/60, 2)
    time_low = round(time*0.7,2)
    time_high = round(time*1.3,2)
  else:
    time = round(((total_input_tokens + total_output_tokens) * 0.01)/60, 2)
    time_low = round(time*0.7,2)
    time_high = round(time*1.3,2)

  quit = False
  print("You are about to annotate", len(text_to_annotate), "text samples and the number of iterations is set to", num_iterations)
  print("Estimated cost range in US Dollars:", cost_low,"-",cost_high)
  print("Estimated minutes to run gpt_annotate():", time_low,"-",time_high)
  print("Please note that these are rough estimates.")
  print("")
  waiting_response = True
  while waiting_response:
    input_response = input("Would you like to proceed and annotate your data? (Options: Y or N) ")
    input_response = str(input_response).lower()
    if input_response == "y" or input_response == "yes":
      waiting_response = False
    elif input_response == "n" or input_response == "no":
      print("")
      print("Exiting gpt_annotate()")
      quit = True
      waiting_response = False
    else:
      print("Please input Y or N.")
  return quit



