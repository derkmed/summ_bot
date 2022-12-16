import argparse
from datetime import date

from datasets import load_from_disk
from enum import Enum

'''
Use this script to transform the summaries extracted from get_summaries.py into
GODEL acceptable format {"CONTEXT", "KNOWLEDGE", "RESPONSE"}

NOTE: Only works for CODS-1!
'''

class Mode(Enum):
    history = 1 # history as "Context"
    summary = 2 # summary as "Context"
    both = 3    # history as "Context" and summary as extra "Knowledge"

def PreprocessForGodel(examples, mode: Mode ):
    dialogs = examples['dialog']
    summaries = examples['summary']
    timesteps = examples['timestep']
    
    contexts = []
    knowledges = []
    responses = []
    for i, d in enumerate(dialogs):
        t = timesteps[i]
        if mode == Mode.summary:
            contexts.append("START {}".format(summaries[i]))
        else:
            contexts.append("START {}".format(' EOS '.join(d[:t])))
            if mode == Mode.both:
                knowledges.append(summaries[i])
        
        responses.append(d[t])
    

    if mode != Mode.both:
        # GODEL requires inputted "Knowledge". But, our project group was lacking it.
        knowledges = ['' for _ in range(len(contexts))]

    return {
        "Context": contexts, 
        "Knowledge": knowledges,
        "Response": responses
        }

def Godel_Mapping_Function(examples):
    return PreprocessForGodel(examples, Mode.history)

def Godel_Summaries_Mapping_Function(examples):
    return PreprocessForGodel(examples, Mode.summary)

def Godel_Both_Mapping_Function(examples):
    return PreprocessForGodel(examples, Mode.both)

if __name__ == "__main__":
    '''
    python3 -i preprocess_for_godel.py
        --cods1 "/home/derekhmd/summ_bot/data/DailySummaryCods1.20221215"
        --cods2 "/home/derekhmd/summ_bot/data/DailySummaryCods2.20221215"
        --codse "/home/derekhmd/summ_bot/data/DailySummaryCodsEnd.20221215"
        --ocods1 "/home/derekhmd/summ_bot/data/GodelSummaryCods1"
        --ocods2 "/home/derekhmd/summ_bot/data/GodelSummaryCods2"
        --ocodse "/home/derekhmd/summ_bot/data/GodelSummaryCodsEnd"
    '''

    today = date.today()
    today = today.strftime("%Y%m%d")
    CODS1_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCods1.{today}"
    CODS2_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCods2.{today}"
    CODS_END_SAVE_PATH = f"/home/derekhmd/summ_bot/data/DailySummaryCodsEnd.{today}"
    OUT_GODEL_PATH = f"/home/derekhmd/summ_bot/data/GodelInput.{today}"
    SUMMARIZED_COLUMN_NAMES = ['timestep', 'dialog', 'summary']
    UNSUMMARIZED_COLUMN_NAMES = ['timestep', 'dialog', 'summary']
    parser = argparse.ArgumentParser(
                    prog = 'json saving of summary augmentaiton dataset',
                    description = 'Obtains daily dialog summary augmentations')
    parser.add_argument('--date', action='store_true')
    parser.add_argument('--cods1', dest='cods1_path', action='store', type=str, default=CODS1_SAVE_PATH,
        help='The input path for 1-sentence summarized random-target dialogs.')
    parser.add_argument('--cods2', dest='cods2_path', action='store', type=str, default=CODS2_SAVE_PATH,
        help='The input path for 2-sentence summarized random-target dialogs.')
    parser.add_argument('--codse', dest='cods_end_path', action='store', type=str, default=CODS_END_SAVE_PATH,
        help='The input path for 1-sentence summarized end-target dialogs.')
    parser.add_argument('--output', dest='out_path', action='store', type=str, default=OUT_GODEL_PATH,
        help='The output directory for GODEL-ready data')
    args = parser.parse_args()

    
    # Handles mapping for raw dialog data.
    out_E_path = f"{args.out_path}/rawE"
    out_1_path = f"{args.out_path}/raw1"
    out_2_path = f"{args.out_path}/raw2"
    augmented_dataE = load_from_disk(args.cods1_path)
    augmented_data1 = load_from_disk(args.cods2_path)
    augmented_data2 = load_from_disk(args.cods_end_path)
    godel_dataE = augmented_dataE.map(Godel_Mapping_Function,
        batched=True, remove_columns=UNSUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_data1 = augmented_data1.map(Godel_Mapping_Function,
        batched=True, remove_columns=UNSUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_data2 = augmented_data2.map(Godel_Mapping_Function,
        batched=True, remove_columns=UNSUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_dataE.save_to_disk(out_E_path)
    godel_data1.save_to_disk(out_1_path)
    godel_data2.save_to_disk(out_2_path)


    # Handles mapping for summary augmented dialog data.
    out_cods_E_path = f"{args.out_path}/codsE"
    out_cods_1_path = f"{args.out_path}/cods1"
    out_cods_2_path = f"{args.out_path}/cods2"
    godel_summary_dataE = augmented_dataE.map(Godel_Summaries_Mapping_Function,
        batched=True, remove_columns=SUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_summary_data1 = augmented_data1.map(Godel_Summaries_Mapping_Function,
        batched=True, remove_columns=SUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_summary_data2 = augmented_data2.map(Godel_Summaries_Mapping_Function,
        batched=True, remove_columns=SUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_summary_dataE.save_to_disk(out_cods_E_path)
    godel_summary_data1.save_to_disk(out_cods_1_path)
    godel_summary_data2.save_to_disk(out_cods_2_path)
    
    # Handles mapping for summary-enriched histories.
    out_both_E_path = f"{args.out_path}/bothE"
    out_both_1_path = f"{args.out_path}/both1"
    out_both_2_path = f"{args.out_path}/both2"
    godel_both_dataE = augmented_dataE.map(Godel_Both_Mapping_Function,
        batched=True, remove_columns=SUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_both_data1 = augmented_data1.map(Godel_Both_Mapping_Function,
        batched=True, remove_columns=SUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_both_data2 = augmented_data2.map(Godel_Both_Mapping_Function,
        batched=True, remove_columns=SUMMARIZED_COLUMN_NAMES, num_proc=4)
    godel_both_dataE.save_to_disk(out_both_E_path)
    godel_both_data1.save_to_disk(out_both_1_path)
    godel_both_data2.save_to_disk(out_both_2_path)
