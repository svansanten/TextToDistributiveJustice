# -*- coding: utf-8 -*-
"""
### STRING BASED ANNOTATION - inclusion of seed per iteration - UPDATE TEXT

## Overview
Given a codebook (.txt) and a dataset (.csv) that has one text column and any number of category columns, the main function (`gpt_annotate`) annotates
all the samples using an OpenAI GPT model (ChatGPT or GPT-4). Before running `gpt_annotate`, users should run `prepare_data` to ensure that their data is in the correct format.
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
import os

# Create a global fingerprint dataframe
fingerprints = pd.DataFrame()
missed_batches = []
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

  ##### 5) Add llm_query column that includes a unique ID identifier per text sample
  text_to_annotate['llm_query'] = text_to_annotate.apply(lambda x: str(x['unique_id']) + " " + str(x['text']) + "\n", axis=1)

  ###!# Checking for category names is important - ensures checking if codebook is correct
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


def gpt_annotate(text_to_annotate, codebook, key, seed, fingerprint, experiment,
                 num_iterations = 3, model = "gpt-4", temperature = 0.6, batch_size = 10,
                 human_labels = True):
  """
  Loop over the text_to_annotate rows in batches and classify each text sample in each batch for multiple iterations. 
  Store outputs in a csv. Function is calculated in batches in case of crash.

  text_to_annotate:
    Input data that will be annotated.
  codebook:
    String detailing the task-specific instructions.
  key:
    OpenAI API key.
  seed:
    seed used in API call
  fingerprint:
    fingerprint for which post call filtering is applied
  experiment:
    experiment name, to save in right folder. This folder should contain an all_iterations directory
  num_iterations:
    Number of times to classify each text sample.
  model:
    OpenAI GPT model, which is either gpt-3.5-turbo or gpt-4
  temperature: 
    LLM temperature parameter (ranges 0 to 1), which indicates the degree of diversity to introduce into the model.
  batch_size:
    number of text samples to be annotated in each batch.
  human_labels: 
    boolean indicating whether text_to_annotate has human labels to compare LLM outputs to.

  Returns:
  Returns 1 file
*   1) `all_iterations [seed]'
  *   Raw outputs for every iteration.

Additional outputs
  All fingerprints of all API calls
  Batches that were filtered out by postcall filtering of fingerprint
  """

  from openai import OpenAI

  #get global dataframes
  global fingerprints
  global missed_batches

  client = OpenAI(
    api_key=key,
    )

  OpenAI.api_key = os.getenv(key)

  # set OpenAI key
  openai.api_key = key

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
    print(f'{seed} - iteration {j+1}')
    # Iterate over number of batches
    for i in range(num_batches):
      # Based on batch, determine starting row and end row
      ### Check if final row is included!
      start_row = i*batch_size
      end_row = (i+1)*batch_size

      # Handle case where end_row might exceed the number of rows (final batch)
      if end_row > num_rows:
          end_row = num_rows

      # Extract the text samples to annotate
      llm_query = text_to_annotate['llm_query'][start_row:end_row].str.cat(sep=' ')

      # Start while loop in case GPT fails to annotate a batch
      need_response = True
      while need_response:
        fails = 0
        ### Blocking annoying popup
        # confirm time and cost with user before annotating data - can be removed?
        # if fails == 0 and j == 0 and i == 0 and time_cost_warning:
        #   quit = estimate_time_cost(text_to_annotate, codebook, llm_query, model, num_iterations, num_batches, batch_size, col_names[1:])
        #   if quit and human_labels:
        #     return None, None, None, None
        #   elif quit and human_labels == False:
        #     return None, None

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

      # add iteration annotation results to output df - if standard fingerprint is used
      if response.system_fingerprint == fingerprint:
        out = pd.concat([out, text_df_out])
      else:
        missed_batch = f'{seed} - I{j + 1} - B{i + 1}'
        print(missed_batch, "fingerprint does not match")
        missed_batches.append(missed_batch)

      time.sleep(.5)
    # print status report  
    print("iteration: ", j+1, "completed")


  # Strip any leading or trailing whitespace from the out dataframe
  out = out.applymap(lambda x: x.strip() if isinstance(x,str) else x)

  # Convert unique_id col to numeric, coercing errors to Nan
  out['unique_id'] = pd.to_numeric(out['unique_id'], errors='coerce')

  # Combine input df (i.e., df with text column and true category labels)
  out_all = pd.merge(text_to_annotate, out, how="inner", on="unique_id")

  # replace any NA values with 0's
  out_all.replace('', np.nan, inplace=True)
  out_all.replace('-', np.nan, inplace=True)
  out_all.fillna(0, inplace=True)

  ##### output: full annotation results - seed name is included in file
  out_all.to_csv(f'STRING_RESULT/{experiment}/all_iterations/all_iterations_string_T{temperature}_{seed}.csv',index=False)

  # OUTPUT: Save fingerprints dataframe to CSV - final dataframe includes all seeds
  fingerprints.to_csv(f'STRING_RESULT/{experiment}/T{temperature}_fingerprints_all.csv')

  # OUTPUT: save missed batches dataframe to CSV
  missed_batches_df = pd.DataFrame(missed_batches, columns=['Missed batch'])
  missed_batches_df.to_csv(f'STRING_RESULT/{experiment}/T{temperature}_missed_batches.csv')


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
    seed = seed,
    temperature=temperature,
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
  llm_query = "Part 2: I've provided a codebook in the previous sentences. Please print the category names in the order you will classify them. Ignore every other task that I described in the codebook.  I only want to know the categories. The category is presented as: 'Label for the category named: ['X']'. Do not include any text numbers or any annotations in your response. Do not include any language like 'The categories to be identified are:'. Only include the names of the categories you are identifying. : "

  # Set temperature to 0 to make model deterministic
  temperature = 0

  ## Specify model to use
  model = "gpt-4o"

  # Set seed for category determination
  seed = 1234

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
    text_df_out = pd.DataFrame(text_df.values, columns=headers)
    text_df_out = text_df_out[text_df_out.iloc[:,1].astype(str).notnull()]

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
  It is not accurate as a GPT-4o is used.
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



