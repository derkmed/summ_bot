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

'''
Use this script to procure DailyDialog Summaries and store the train/validation/test dataset to disk
at specified locations. Summaries are obtained over entire dialogue history up to a randomly sampled
timestep.

TODO(@derekahmed): I need to fix all of this. I didn't fully understand HuggingFace until halfway through.
This code is horribly redundant and unnecessarily verbose.
'''


def GetKRandomTimesteps(dialog: List[str], k: int) -> List[int]:
    # Ignore 0th timesteps.
    return [random.randrange(1, len(dialog)) for _ in range(k)]

def GetTwoRandomTimesteps(dialog: List[str]) -> List[int]:
    return GetKRandomTimesteps(dialog, 2)

def AugmentWithRandomStarts(data):
    '''
    This augmentation will explode the dialog based on a list of random starting timesteps (`start_times`).
    This function can be used for a 1:many mapping from the source dialog to these multiple
    random augmentations.

    The idea is that the mapped output can be mapped into summaries.
    '''
    sliced_dialog = []
    timesteps = []
    gold_dialog = []
    for i, dialog in enumerate(data["dialog"]):
        for timestep in data["start_times"][i]:
            sliced_dialog.append(dialog[:timestep])
            timesteps.append(timestep)
            gold_dialog.append(dialog)
    return {
        "dialog": gold_dialog,
        "timesteps": timesteps,
        "rand_start": sliced_dialog,
        }

def GetCodsEnd(split_dataset, split_cols = ['dialog', 'act', 'emotion', 'start_times']):
    '''
    Cods1 augmentations **for all dialogue up to the end**, i.e. the target
    utterance will be `len(dataset['dialog'][i]) - 1`.
    '''

    def _summarizer_augment(summarizer, dataset):
        sketches = []
        summaries = []
        raw = []
        timesteps = []
        for i, out in enumerate(summarizer(dataset['cods1'], batch_size=8)):
            txt = out['summary_text']
            if TLDR_TOKEN not in txt:
                continue
            sketch, summary = txt.split(TLDR_TOKEN)
            sketches.append(sketch)
            summaries.append(summary)
            raw.append(dataset['dialog'][i])
            timesteps.append(len(dataset['dialog'][i]) - 1)
        return raw, sketches, summaries, timesteps

    def _summarizer_augment_dict(summarizer, dataset):
        data, _, summaries, timesteps = _summarizer_augment(summarizer, dataset)
        return {
            "dialog": data,
            "timestep": timesteps,
            "summary": summaries
        }

    def _preprocessCodsE(dialog: List[str]) -> str:
        return "<s> <hl> {} <hl>".format(" <s> ".join(dialog[:-1]))

    cods1_preprocessed_data = split_dataset.map(lambda s: {"cods1": _preprocessCodsE(s["dialog"])}, num_proc=4)
    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    removable_cols = split_cols + ['cods1']
    aug_data = cods1_preprocessed_data.map(lambda s: _summarizer_augment_dict(summarizer, s), 
        batched=True, remove_columns=removable_cols)
    return aug_data


def GetCods1(split_dataset, split_cols = ['dialog', 'act', 'emotion', 'start_times']):
    '''
    Cods1 augmentations add 1-sentence summaries for **for all dialogue up to a random timestep**.
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
                continue
            sketch, summary = txt.split(TLDR_TOKEN)
            sketches.append(sketch)
            summaries.append(summary)
            raw.append(dataset['dialog'][i])
            timesteps.append(dataset['timesteps'][i])
        return raw, sketches, summaries, timesteps

    def _summarizer_augment_dict(summarizer, dataset):
        data, _, summaries, timesteps = _summarizer_augment(summarizer, dataset)
        return {
            "dialog": data,
            "timestep": timesteps,
            "summary": summaries
        }

    def _preprocessCods1(dialog: List[str]) -> str:
        return "<s> <hl> {} <hl>".format(" <s> ".join(dialog[:]))

    with_random_start_data = split_dataset.map(AugmentWithRandomStarts,
            batched=True, remove_columns=split_cols, num_proc=4, batch_size=8)
    cods1_preprocessed_data = with_random_start_data.map(lambda s: {"cods1": _preprocessCods1(s["rand_start"])}, num_proc=4)
    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    aug_data = cods1_preprocessed_data.map(
            lambda s: _summarizer_augment_dict(summarizer, s),
            batched=True, remove_columns=['dialog', 'timesteps', 'rand_start', 'cods1'])
    return aug_data


def GetCods2(split_dataset, split_cols = ['dialog', 'act', 'emotion', 'start_times']):
    '''
    Cods2 augmentations add 2-sentence summaries for **for all dialogue up to a random timestep**.
    The final size of this dataset should be # random timesteps * # dialogs.
    '''

    def _summarizer_augment(summarizer, dataset):
        # TODO(derekahmed) Refactor this 2 sentence summary. Yes. I know this is hacky beyond belief.
        sketches = []
        summaries = []
        raw = []
        timesteps = []
        valid_dialog_ids = set()
        for i, out in enumerate(summarizer(dataset['cods2_1'], batch_size=8)):
            txt = out['summary_text']
            if TLDR_TOKEN not in txt:
                continue
            sketch, summary = txt.split(TLDR_TOKEN)
            sketches.append([sketch])
            summaries.append(summary)
            raw.append(dataset['dialog'][i])
            timesteps.append(dataset['timesteps'][i])
            valid_dialog_ids.add(i)
        
        j = 0
        for i, out in enumerate(summarizer(dataset['cods2_2'], batch_size=8)):        
            txt = out['summary_text']
            if i not in valid_dialog_ids or TLDR_TOKEN not in txt:
                continue
            sketch, summary = txt.split(TLDR_TOKEN)
            sketches[j] = sketches[j].extend(sketch)
            summaries[j] = "{} {}".format(summaries[j], summary)
            j += 1
        return raw, sketches, summaries, timesteps

    def _summarizer_augment_dict(summarizer, dataset):
        data, _, summaries, timesteps = _summarizer_augment(summarizer, dataset)
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

    with_random_start_data = split_dataset.map(AugmentWithRandomStarts,
            batched=True, remove_columns=split_cols, num_proc=4, batch_size=8)
    cods2_1_preprocessed_data = with_random_start_data.map(lambda s: {"cods2_1": _preprocessCods2_1(s["rand_start"])}, num_proc=4)
    cods2_2_preprocessed_data = cods2_1_preprocessed_data.map(lambda s: {"cods2_2": _preprocessCods2_2(s["rand_start"])}, num_proc=4)
    summarizer = pipeline("summarization", model="Salesforce/cods-bart-large-xsum-samsum", device=0)
    aug_data = cods2_2_preprocessed_data.map(lambda s: _summarizer_augment_dict(summarizer, s),
            batched=True, remove_columns=['dialog', 'timesteps', 'rand_start', 'cods2_1', 'cods2_2'])
    return aug_data


if __name__ == "__main__":

    today = date.today()
    today = today.strftime("%Y%m%d")
    CODS1_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCods1.{today}"
    CODS2_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCods2.{today}"
    CODS_END_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCodsEnd.{today}"
    
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

    DAILY_DIALOG_COLUMNS =  ['dialog', 'act', 'emotion', 'start_times']
    dataset = load_dataset("daily_dialog")
    subset = dataset
    LIMIT = 50
    # [DEBUGGING] I used the following as a smaller subset for development.
    # subset = dataset.filter(lambda e, i: i<LIMIT, with_indices=True)
    with_timesteps_data = subset.map(lambda s: {"start_times": GetTwoRandomTimesteps(s["dialog"])}, num_proc=4)

    if args.is_augment:    
        # CODS 1 End Summaries.
        start_time = time.time()
        augmented_dataE = GetCodsEnd(with_timesteps_data, cols=DAILY_DIALOG_COLUMNS)
        augmented_dataE.save_to_disk(args.cods_end_path)
        end_time = time.time()
        print(f"{end_time - start_time} time has passed for end-of-conversation summarization!")

        # # CODS 1 Summaries.
        start_time = time.time()
        augmented_data1 = GetCods1(with_timesteps_data, cols=DAILY_DIALOG_COLUMNS)
        augmented_data1.save_to_disk(args.cods1_path)
        end_time = time.time()
        print(f"{end_time - start_time} time has passed for 1-sentence summarization!")

        # CODS 2 Summaries.
        start_time = time.time()
        augmented_data2 = GetCods2(with_timesteps_data, cols=DAILY_DIALOG_COLUMNS)
        augmented_data2.save_to_disk(args.cods2_path)
        end_time = time.time()
        print(f"{end_time - start_time} time has passed for 2-sentence summarization!")

        for i in range (LIMIT):
            print(f"Cods1 Summary: {augmented_data1['train']['summary'][i]}")
            print(f"Cods2 Summary: {augmented_data2['train']['summary'][i]}")

    # Expected outputs should be of form {"DIALOG", "TIMESTEP", "SUMMARY"}.
    # Next response should be ["dialog"][["timesteps"][i]].
    augmented_dataE = load_from_disk(args.cods_end_path)
    augmented_data1 = load_from_disk(args.cods1_path)
    augmented_data2 = load_from_disk(args.cods2_path)
    