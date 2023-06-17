'@zklsmr'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import argparse
import PyPDF2
from transformers import pipeline
from tqdm import tqdm


logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow.stream_executor").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.captureWarnings(True)


parser = argparse.ArgumentParser()
parser.add_argument("pdf_in", help="PDF file to summarize")
parser.add_argument("txt_out", help="output file")
parser.add_argument("max_l", help="max length", default=250)
parser.add_argument("min_l", help="min length", default=50)
parser.add_argument("chunk_arg",  help="chunk proportion from total text", default=0.25)
args = parser.parse_args()


print("==============================================================================")
print("Summarization will commence after parcellation and tokenization of the article")


summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")


#other models that don't work for now (from facebook and google). Something for the dev branch
#Summarizer = pipeline("summarization",  model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", framework="tf")
#Summarizer = pipeline("summarization",model="google/pegasus-xsum", tokenizer="google/pegasus-xsum")


with open(args.pdf_in, 'rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    num_pages = len(pdf_reader.pages)
    all_text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text:
          all_text += text

print("==============================================================================")
nChunk = int(len(all_text)*float(args.chunk_arg))
chunks = [all_text[i:i+nChunk] for i in range(0, len(all_text), nChunk)]
print(f"based on the {args.chunk_arg} proportion there are {len(chunks)} chunks")


all_summary = []
for _ch in tqdm(chunks):
  summary = summarizer(_ch, max_length=int(args.max_l), min_length=int(args.min_l), do_sample=False)
  all_summary.append(summary[0]["summary_text"])

print("==============================================================================")
print(f"Saving to {args.txt_out}")
with open(args.txt_out, "w") as o:
  for line in all_summary:
    o.write(f"{line}\n")
print("==============================================================================")
print("done")
print("==============================================================================")


