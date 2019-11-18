from demo.label_wav import load_labels, load_graph, run_graph
import tensorflow as tf
import argparse
import glob
import os
import sys

FLAGS = None

def load_graph_and_label(labels, graph):
    if not labels or not tf.io.gfile.exists(labels):
        tf.compat.v1.logging.fatal('Labels file does not exist %s', labels)

    if not graph or not tf.io.gfile.exists(graph):
        tf.compat.v1.logging.fatal('Graph file does not exist %s', graph)

    labels_list = load_labels(labels)

    # load graph, which is stored in the default session
    load_graph(graph)

    return labels_list


def label_wav(wav, labels_list, input_name, output_name, how_many_labels):
    """Loads the model and labels, and runs the inference to print predictions."""
    if not wav or not tf.io.gfile.exists(wav):
        tf.compat.v1.logging.fatal('Audio file does not exist %s', wav)

    with open(wav, 'rb') as wav_file:
        wav_data = wav_file.read()

    label, score = run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
    return label, score


def main(_):
    """Entry point for script, converts flags to arguments."""
    labels_list = load_graph_and_label(FLAGS.labels, FLAGS.graph)

    wav_paths = glob.glob(os.path.join(FLAGS.wav_folder, "*.wav"))

    for wav in wav_paths:
        label, score = label_wav(wav, labels_list, FLAGS.input_name,
                  FLAGS.output_name, FLAGS.how_many_labels)

        print(f"{wav} \t {label}: {score}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav_folder', type=str, default='', help='folder of audio files to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

