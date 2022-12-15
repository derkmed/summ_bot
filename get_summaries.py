import argparse
import random
import time

from typing import List

from tqdm import tqdm

from datasets import load_dataset, load_from_disk
from datasets import Dataset
from datasets.dataset_dict import DatasetDict

from transformers import pipeline, pipelines

from datetime import date

TLDR_TOKEN = "TLDR" # " TLDR "?

def GetRandomTimesteps(dialog: List[str], t: int) -> List[int]:
    # Ignore 0th timesteps.
    return [random.randrange(1, len(dialog))]

def GetTwoRandomTimesteps(dialog: List[str]) -> List[int]:
    # Ignore 0th timesteps.
    return [random.randrange(1, len(dialog)), random.randrange(0, len(dialog))]

def batched_random_start_augment(data):
    '''
    This augmentation is meant to be called with
    `.map(augment_data_batched, batched=True, remove_columns=...column_names, batch_size=...)`
    It will explode all randomly-starting dialogs and output these with their respective random starts and total # timesteps.
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

def GetCodsEnd(split_dataset):
    '''
    Cods1 augmentations are defined here without random start.
    '''

    def _summarizer_augment(summarizer, dataset):
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
            timesteps.append(len(dataset['dialog'][i]) - 1)
        return raw, sketches, summaries, timesteps

    def _summarizer_augment_dict(summarizer, dataset):
        data, sketches, summaries, timesteps = _summarizer_augment(summarizer, dataset)
        return {
            "dialog": data,
            "timestep": timesteps,
            "summary": summaries
        }

    def _preprocessCodsE(dialog: List[str]) -> str:
        return "<s> <hl> {} <hl>".format(" <s> ".join(dialog[:-1]))

    cods1_preprocessed_data = split_dataset.map(lambda s: {"cods1": _preprocessCodsE(s["dialog"])}, num_proc=4)
    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    aug_data = cods1_preprocessed_data.map(
            lambda s: _summarizer_augment_dict(summarizer, s),
            batched=True,
            remove_columns=['start_times', 'dialog', 'act', 'emotion', 'cods1'])
    return aug_data

def GetCods1(split_dataset):
    '''
    Cods1 augmentations are defined here.
    Important note: we want to 1:many map each dialog to multiple dialogs, *while* remove existing columns.
    Doing so requires a lot of care with respect to columns and only works at the train/validation/test split level.
    The final size of this dataset should be # random timesteps * # dialogs.
    '''

    def _summarizer_augment(summarizer, dataset):
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

    def _summarizer_augment_dict(summarizer, dataset):
        data, sketches, summaries, timesteps = _summarizer_augment(summarizer, dataset)
        return {
            "dialog": data,
            "timestep": timesteps,
            "summary": summaries
        }

    def _preprocessCods1(dialog: List[str]) -> str:
        return "<s> <hl> {} <hl>".format(" <s> ".join(dialog[:]))

    with_random_start_data = split_dataset.map(batched_random_start_augment,
            batched=True,
            remove_columns=['start_times', 'dialog', 'act', 'emotion'],
            num_proc=4, batch_size=8)
    cods1_preprocessed_data = with_random_start_data.map(
        lambda s: {"cods1": _preprocessCods1(s["rand_start"])}, num_proc=4)
    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    aug_data = cods1_preprocessed_data.map(
            lambda s: _summarizer_augment_dict(summarizer, s),
            batched=True,
            remove_columns=['dialog', 'timesteps', 'rand_start', 'cods1'])
    return aug_data

def GetCods2(split_dataset):
    '''
    Cods2 augmentations are defined here.
    Important note: we want to 1:many map each dialog to multiple dialogs, *while* remove existing columns.
    Doing so requires a lot of care with respect to columns and only works at the train/validation/test split level.
    The final size of this dataset should be # random timesteps * # dialogs.
    '''

    def _summarizer_augment(summarizer, dataset):
        sketches = []
        summaries = []
        raw = []
        timesteps = []
        dialog_is = set()
        for i, out in enumerate(summarizer(dataset['cods2_1'], batch_size=8)):
            # Handle first sentence summary.
            txt = out['summary_text']
            if TLDR_TOKEN not in txt:
                # We can't do anything so skip this sample.
                continue
            sketch, summary = txt.split(TLDR_TOKEN)
            sketches.append([sketch])
            summaries.append(summary)
            raw.append(dataset['dialog'][i])
            timesteps.append(dataset['timesteps'][i])
            dialog_is.add(i)
        
        j = 0
        for i, out in enumerate(summarizer(dataset['cods2_2'], batch_size=8)):        
            # Handle second sentence summary.
            txt = out['summary_text']
            if i not in dialog_is or TLDR_TOKEN not in txt:
                continue
            sketch, summary = txt.split(TLDR_TOKEN)
            sketches[j] = sketches[j].extend(sketch)
            summaries[j] = "{} {}".format(summaries[j], summary)
            j += 1
        return raw, sketches, summaries, timesteps

    def _summarizer_augment_dict(summarizer, dataset):
        data, sketches, summaries, timesteps = _summarizer_augment(summarizer, dataset)
        return {
            "dialog": data,
            "timestep": timesteps,
            "summary": summaries
        }

    def _preprocessCods2_1(dialog: List[str]) -> str:
        h = int(len(dialog) / 2)
        return "<s> <hl> {} <hl> {}".format(" <s> ".join(dialog[:h]), " <s> ".join(dialog[h:]))

    def _preprocessCods2_2(dialog: List[str]) -> str:
        h = int(len(dialog) / 2)
        return "<s> {} <hl> {} <hl>".format(" <s> ".join(dialog[:h]), " <s> ".join(dialog[h:]))

    with_random_start_data = split_dataset.map(batched_random_start_augment,
            batched=True, 
            remove_columns=['start_times', 'dialog', 'act', 'emotion'],
            num_proc=4, batch_size=8)
    cods2_1_preprocessed_data = with_random_start_data.map(lambda s: {"cods2_1": _preprocessCods2_1(s["rand_start"])}, num_proc=4)
    cods2_2_preprocessed_data = cods2_1_preprocessed_data.map(lambda s: {"cods2_2": _preprocessCods2_2(s["rand_start"])}, num_proc=4)

    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    aug_data = cods2_2_preprocessed_data.map(
            lambda s: _summarizer_augment_dict(summarizer, s),
            batched=True,
            remove_columns=['dialog', 'timesteps', 'rand_start', 'cods2_1', 'cods2_2'])
    return aug_data


if __name__ == "__main__":

    today = date.today()
    today = today.strftime("%Y%m%d")
    CODS1_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCods1.{today}"
    CODS2_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCods2.{today}"
    CODS_END_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCodsEnd.{today}"

    AUGMENT_DATA = False

    parser = argparse.ArgumentParser(
                    prog = 'summarization_augmentation',
                    description = 'Obtains daily dialog summary augmentations')

    parser.add_argument('--augment', dest='is_augment', action='store_true')
    parser.add_argument('--date', action='store_true')
    parser.add_argument('--cods1', dest='cods1_path', action='store', type=str, default=CODS1_SAVE_PATH,
        help='The path for 1-sentence summarized random-target dialogs.')
    parser.add_argument('--cods2', dest='cods2_path', action='store', type=str, default=CODS2_SAVE_PATH,
        help='The path for 2-sentence summarized random-target dialogs.')
    parser.add_argument('--codse', dest='cods_end_path', action='store', type=str, default=CODS_END_SAVE_PATH,
        help='The path for 1-sentence summarized end-target dialogs.')


    args = parser.parse_args()

    daily_dialog_columns =  ['dialog', 'act', 'emotion']
    dataset = load_dataset("daily_dialog")

    if args.is_augment:
        subset = dataset
        LIMIT = 50
        subset = dataset.filter(lambda e, i: i<LIMIT, with_indices=True)
        with_timesteps_data = subset.map(
            lambda s: {"start_times": GetTwoRandomTimesteps(s["dialog"])}, num_proc=4)
        # daily_dialog_columns =  ['dialog', 'act', 'emotion', 'start_times']
        
        # TODO: The following unnecessarily splits train/test/validation. I didn't know HuggingFace at the time.
        # CODS 1 End Summaries.
        start_time = time.time()
        augmented_dataE = GetCodsEnd(with_timesteps_data)
        augmented_dataE.save_to_disk(args.cods_end_path)
        end_time = time.time()
        print(f"{end_time - start_time} time has passed!")

        # # CODS 1 Summaries.
        start_time = time.time()
        augmented_data1 = GetCods1(with_timesteps_data)
        augmented_data1.save_to_disk(args.cods1_path)
        end_time = time.time()
        print(f"{end_time - start_time} time has passed!")

        # CODS 2 Summaries.
        start_time = time.time()
        augmented_data2 = GetCods2(with_timesteps_data)
        augmented_data2.save_to_disk(args.cods2_path)
        end_time = time.time()
        print(f"{end_time - start_time} time has passed!")


        for i in range (LIMIT):
            # print(f"Cods1 Summary: {augmented_data1['train']['summary'][i]}")
            print(f"Cods2 Summary: {augmented_data2['train']['summary'][i]}")

    augmented_dataE = load_from_disk(args.cods_end_path)
    augmented_data1 = load_from_disk(args.cods1_path)
    augmented_data2 = load_from_disk(args.cods2_path)
    