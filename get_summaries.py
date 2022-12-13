import random


from datasets import load_dataset
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
from transformers import BlenderbotConfig, BlenderbotForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, pipelines

from typing import List

import torch

from tqdm import tqdm

def GetRandomTimesteps(dialog: List[str], t: int) -> List[int]:
    return [random.randrange(0, len(dialog))]

def GetTwoRandomTimesteps(dialog: List[str]) -> List[int]:
    return [random.randrange(0, len(dialog)), random.randrange(0, len(dialog))]

def PreProcessCods1(dialog: List[str]) -> str:
    return "<s> <hl> {} <hl>".format(" <s> ".join(dialog))

def GetOneSentenceSummary(summarizer_pipeline: pipelines.text2text_generation.SummarizationPipeline, 
                data: List[str]):
    '''
    Iterate through `data` using `summarizer_pipeline` to generate summaries.
    '''
    batch_sketches = []
    batch_summaries = []
    for i, out in enumerate(summarizer_pipeline(batch, batch_size=8)):
        
        txt = out['summary_text']
        if "TLDR" not in txt:
            continue
        sketch, summary = txt.split(" TLDR ")
        batch_sketches.append(sketch)
        batch_summaries.append(summary)
    return batch_sketches, batch_summaries    


def random_start_augment(data):
    '''
    This augmentation is meant to be called with
    `.map(..., num_proc=4)`
    '''
    outputs = []
    for s in data["start_times"]:
        print(data["dialog"])
        outputs.append(data["dialog"][:s])
    print(outputs)
    return {"data": outputs}

def batched_random_start_augment(data):
    '''
    This augmentation is meant to be called with
    `.map(augment_data_batched, batched=True, remove_columns=...column_names, batch_size=...)`
    '''
    sliced = []
    timesteps = []
    dialogs = []
    for i, dialog in enumerate(data["dialog"]):
        for timestep in data["start_times"][i]:
            sliced.append(dialog[:timestep])
            timesteps.append(timestep)
            dialogs.append(dialog)
    return {
        "dialog": dialogs,
        "timesteps": timesteps,
        "rand_start": sliced,
        }

def summarizer_augment(summarizer, dataset):
    TLDR_TOKEN = "TLDR" # " TLDR "?
    sketches = []
    summaries = []
    raw = []
    timesteps = []
    for i, out in enumerate(summarizer(dataset['cods1'], batch_size=8)):
        txt = out['summary_text']
        if TLDR_TOKEN not in txt:
            # We can't do anything so skip this sample.
            continue
        sketch, summary = txt.split(TLDR_TOKEN)
        sketches.append(sketch)
        summaries.append(summary)
        raw.append(dataset['dialog'][i])
        timesteps.append(dataset['timesteps'][i])
    return raw, sketches, summaries, timesteps

def summarizer_augment_dict(summarizer, dataset):
    data, sketches, summaries, timesteps = summarizer_augment(summarizer, dataset)
    return {
        "data": data,
        "sketch": sketches,
        "summary": summaries,
        "timesteps": timesteps
    }

if __name__ == "__main__":
    # Daily dialog Dataset loading & tokenization
    # dataset = load_dataset("yelp_review_full")
    dataset = load_dataset("daily_dialog")
    training_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']
    
    LIMIT = 1000
    subset = dataset.filter(lambda e, i: i<LIMIT, with_indices=True)
    with_timesteps_data = subset.map(lambda s: {"start_times": GetTwoRandomTimesteps(s["dialog"])}, num_proc=4)
    # random_start_data = with_timesteps_data.map(random_start_augment, num_proc=4)
    
    # In this example, we want to 1:many map each dialog to multiple dialogs, *while* remove existing columns.
    # Doing so requires a lot of care with respect to columns and only works at the train/validation/test split level.
    # The final size of this dataset should be # random timesteps * # dialogs.
    training_with_starts_data = with_timesteps_data['train']
    erandom_start_data = training_with_starts_data.map(batched_random_start_augment,
        batched=True, remove_columns=training_with_starts_data.column_names, num_proc=4, batch_size=8)
    cods1_preprocessed_data = erandom_start_data.map(lambda s: {"cods1": PreProcessCods1(s["rand_start"])}, num_proc=4)
    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    aug_data = cods1_preprocessed_data.map(lambda s: summarizer_augment_dict(summarizer, s),
        batched=True, remove_columns=cods1_preprocessed_data.column_names)

    # need raw, timestep, and summary
