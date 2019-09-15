import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from pydub import AudioSegment

import os
import tqdm


def join_action(row, index):
    if np.isnan(row[f"start_voc_{index}"]):
        return None
    else:
        return ",".join(row[f"type_voc_{index},start_voc_{index},end_voc_{index}".split(",")].astype(str))


def expand_actions(row):
    return row["value"].split(",")


def get_labels(labels_path):
    labels = pd.read_csv(labels_path)
    for i in range(1, 7):
        labels[f"action_{i}"] = labels.apply(join_action, axis=1, args=(i,))

    labels = labels[['Sample', 'original_spk', 'gender', 'original_time', 'action_1', 'action_2', 'action_3',
                     'action_4', 'action_5', 'action_6']]

    labels = labels.melt(['Sample', 'original_spk', 'gender', 'original_time'])

    labels = labels.dropna()

    labels = pd.concat([labels, labels.apply(lambda x: pd.Series(x["value"].split(",")), axis=1)], axis=1)
    labels = labels[['Sample', 'original_spk', 'gender', 'original_time', 0, 1, 2]]
    labels.columns = ['Sample', 'original_spk', 'gender', 'original_time', "type", "start", "end"]

    labels = labels[labels.type == "filler"]
    labels["start"] = labels.start.astype(float)
    labels["end"] = labels.end.astype(float)
    return labels


def get_2s_label(row, filtered_sample_df):
    for sample_row in filtered_sample_df.iterrows():
        if sample_row[1]['end'] < row['start'] or sample_row[1]['start'] > row['end']:
            continue
        elif sample_row[1]['start'] > row['start'] and sample_row[1]['end'] < row['end']:
            return 1
        else:
            if sample_row[1]['start'] > row['start']:
                overlap = row['end'] - sample_row[1]['start']
            else:
                overlap = sample_row[1]['end'] - row['start']

            if overlap > 1 or (overlap / (sample_row[1]['end'] - sample_row[1]['start'])) > 0.9:
                return 1

            # TODO: what to do when somewhere in between

    return 0


def gen_2s_sections(labels_df):
    samples = labels_df["Sample"].unique()
    print("labling 2 second clips")
    for sample in tqdm.tqdm(samples):
        section_df = pd.DataFrame(np.linspace(0, 11, 23))
        section_df.columns = ['start']
        section_df['end'] = section_df.shift(-4)
        section_df = section_df.dropna()

        section_df["name"] = [sample + "-" + str(i) for i in range(len(section_df))]
        section_df["sample"] = sample

        section_df["original_spk"] = labels_df[labels_df.Sample == sample]["original_spk"].values[0]

        section_df["is_filler"] = section_df.apply(get_2s_label, axis=1, args=(labels_df[labels_df.Sample == sample], ))

        yield section_df


def train_test_split(labeled_sections):
    # split people
    speakers = labeled_sections.original_spk.unique()
    train_speakers, val_speakers = sklearn_train_test_split(speakers, test_size=0.1)

    # split samples
    train_samples, val_samples = sklearn_train_test_split(
        labeled_sections[labeled_sections.original_spk.isin(train_speakers)], test_size=0.15)

    def label_train_val(row):
        nonlocal val_samples
        nonlocal val_speakers

        if row['original_spk'] in val_speakers or row['sample'] in val_samples:
            return "val"
        else:
            return "train"

    labeled_sections['set'] = labeled_sections.apply(label_train_val, axis=1)


def split_save_audio(labeled_sections):
    if not os.path.exists("datasets/vocalizationcorpus/data_2s"):
        os.mkdir("datasets/vocalizationcorpus/data_2s")

    if not os.path.exists("datasets/vocalizationcorpus/data_2s/val"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/val")
    if not os.path.exists("datasets/vocalizationcorpus/data_2s/val/um"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/val/um")
    if not os.path.exists("datasets/vocalizationcorpus/data_2s/val/other"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/val/other")

    if not os.path.exists("datasets/vocalizationcorpus/data_2s/train"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/train")
    if not os.path.exists("datasets/vocalizationcorpus/data_2s/train/um"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/train/um")
    if not os.path.exists("datasets/vocalizationcorpus/data_2s/train/other"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/train/other")

    print("saving_segments:")
    for sample, data in tqdm.tqdm(labeled_sections.groupby("sample")):
        orig_file_path = f"datasets/vocalizationcorpus/data/{sample}.wav"

        audio_sample = AudioSegment.from_wav(orig_file_path)

        for row in data.iterrows():
            start = row[1]['start']
            end = row[1]['end']

            if row[1]["is_filler"]:
                is_um = "um"

            else:
                is_um = "other"

            segment_path = f"datasets/vocalizationcorpus/data_2s/{row[1]['set']}/{is_um}/{row[1]['name']}.wav"

            sample_segment = audio_sample[start * 1000: end * 1000]
            sample_segment.export(segment_path, format="wav", )


if __name__ == "__main__":

    if os.path.exists("labels.pickle"):
        labels_df = pd.read_pickle("labels.pickle")
    else:
        print("getting labels")
        labels_df = get_labels("datasets/vocalizationcorpus/labels.txt")
        labels_df.to_pickle("labels.pickle")

    if os.path.exists("labeled_sections.pickle"):
        labeled_sections = pd.read_pickle("labeled_sections.pickle")
    else:
        labeled_sections = pd.concat(gen_2s_sections(labels_df))
        labeled_sections.to_pickle("labeled_sections.pickle")

    train_test_split(labeled_sections)
    labeled_sections.to_pickle("labeled_sections_splitted.pickle")

    split_save_audio(labeled_sections)



