# TextToDistributiveJustice
*TU Delft Masterthesis - Aiming to identify Distributive Justice Principles in COP high level segment speeches* 

This repository was used in a masterthesis that aimed to answer the following question: 
"To what extent can Large Language Models accurately identify and enhance our understanding of distributive justice preferences in climate negotiation texts?"
This question was answered with a case study of COP High Level Segment speeches and GPT-4o. The complete research is available at: [repository]

**Workflow**
The workflow and software for LLM-augmented text annotation is adapted from [Pangakis et al. (2023)](https://arxiv.org/abs/2306.00176).
Part A: Manual annotation
A deductive approach is used to create a theoretical codebook to classify sentences for distributive justice principles. In addition to principles, sentences are annotated for relevance, topic, unit of distribution, and shape of distibution. 
The codebook is updated during manual annotation of a subset of 51 speeches. Annotated speeches are used as ground truth dataset. 
Part B: LLM annotation
GPT-4o is tasked with classifying sentences using the updated codebooks. 
The model’s performance is compared to the ground truth dataset with performance metrics such as accuracy, precision, recall, and F1 score.

In addition to the results generated in this study, this repository presents a corpus of HLS speeches of COP19-COP28 in both PDF and TXT documents.

**Directories**
- HLS CORPUS
  - 1083 PDF documents, split into English and non english
  - 742 TXT documents, converted English PDF 
  - HLS availability COP19-COP28, Excel file with an overview of all available speeches per cop
  - HLS_pdf_2txt.ipynb, notebook used to parse PDF to text
- HLS subset - Manual annotation
  - Raw manual annotation files including short motivations for appointed labels per sentence
  - HLS_full.csv, combined manually annotated dataset with preprocessed numerical and string labels per category
  - Manual annotation - Results.ipynb, exploratory data analysis of HLS_full
  - Manual annotation - numerical dataprep & string dataprep, conversion of raw annotation files to pre-defined categories
- DJ_GPT_analysis (Additional ReadME)
  - Codebooks
  - data
  - results
  - gpt_annotate_num.py
  - gpt_annotate_string.py
- Result analysis 

Reach out to suze.vansanten@gmail.com with any questions or suggestions. 