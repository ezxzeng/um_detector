import os
import shutil
import pandas as pd

import tqdm
tqdm.tqdm.pandas()


def copy_with_new_name(row):
    if row["is_filler"]:
        is_um = "um"
    else:
        is_um = "other"

    segment_path = f"datasets/vocalizationcorpus/data_2s/{row['set']}/{is_um}/{row['name']}.wav"

    new_path = f"datasets/vocalizationcorpus/data_2s/reformatted/{is_um}/{row['original_spk']}-{row['sample']}-nohash-{row['name'][-1]}-.wav"

    shutil.copy(segment_path, new_path)


if __name__ == "__main__":
    if not os.path.exists("datasets/vocalizationcorpus/data_2s/reformatted"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/reformatted")

    if not os.path.exists("datasets/vocalizationcorpus/data_2s/reformatted/other"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/reformatted/other")

    if not os.path.exists("datasets/vocalizationcorpus/data_2s/reformatted/um"):
        os.mkdir("datasets/vocalizationcorpus/data_2s/reformatted/um")

    labels_sections_splitted = pd.read_pickle("datasets/vocalizationcorpus/labeled_sections_splitted.pickle")
    labels_sections_splitted.progress_apply(copy_with_new_name, axis=1)