import tensorflow as tf 
import tensorflow_federated as tff 
import tensorflow_datasets as tfds 
import collections
import numpy as np
import attr
from typing import Callable, Optional, Union
from matplotlib import pyplot as plt


# Construct dataset: every 
num_clients = 10
client_ids = [str(i) for i in range(num_clients)]
train_ds = tfds.load('mnist', split='train')
assert isinstance(train_ds, tf.data.Dataset)
train_ds = train_ds.shuffle(60000, seed = 1, reshuffle_each_iteration=False).take(6000)
lengt_dataset = train_ds.reduce(0, lambda x,_: x+1).numpy()
print('Length of dataset', lengt_dataset)
def client_id_to_dataset(id:str) -> tf.data.Dataset:
    sharded_ds = train_ds.filter(lambda x: x['label']== int(id)).take(500)
    assert isinstance(sharded_ds, tf.data.Dataset)
    return sharded_ds

emnist_train = tff.simulation.ClientData.from_clients_and_fn(client_ids, client_id_to_dataset)
example_dataset = emnist_train.create_tf_dataset_for_client(
     emnist_train.client_ids[0])
dataset_length = 500


############# VERIFY DATA #################################
# example_dataset = emnist_train.create_tf_dataset_for_client(
#     emnist_train.client_ids[0])

# example_element = next(iter(example_dataset))
# plt.imshow(example_element['image'].numpy(), cmap='gray', aspect='equal')
# plt.grid(False)
# plt.savefig('/jet/home/houc/federated/robust_het/exfig.jpg')
# plt.close()
# # Number of examples per layer for a sample of clients
# f = plt.figure(figsize=(12, 7))
# f.suptitle('Label Counts for a Sample of Clients')
# for i in range(6):
#   client_dataset = emnist_train.create_tf_dataset_for_client(
#       emnist_train.client_ids[i])
#   plot_data = collections.defaultdict(list)
#   for example in client_dataset:
#     # Append counts individually per label to make plots
#     # more colorful instead of one color per plot.
#     label = example['label'].numpy()
#     plot_data[label].append(label)
#   plt.subplot(2, 3, i+1)
#   plt.title('Client {}'.format(i))
#   for j in range(10):
#     plt.hist(
#         plot_data[j],
#         density=False,
#         bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.savefig('/jet/home/houc/federated/robust_het/histogram.jpg')
# plt.close()
# for i in range(5):
#   client_dataset = emnist_train.create_tf_dataset_for_client(
#       emnist_train.client_ids[i])
#   plot_data = collections.defaultdict(list)
#   for example in client_dataset:
#     plot_data[example['label'].numpy()].append(example['image'].numpy())
#   f = plt.figure(i, figsize=(12, 5))
#   f.suptitle("Client #{}'s Mean Image Per Label".format(i))
#   for j in range(10):
#     mean_img = np.mean(plot_data[j], 0)
#     plt.subplot(2, 5, j+1)
#     plt.imshow(mean_img.reshape((28, 28)))
#     plt.axis('off')
#   plt.savefig('/jet/home/houc/federated/robust_het/averages_{0}.jpg'.format(i))
#   plt.close()

############# VERIFY DATA #################################
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

NUM_CLIENTS = 10
### number of rounds * number of client steps
NUM_EPOCHS = 5
BATCH_SIZE = 600
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset, client_id):

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['image'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).batch(1).map(batch_format_fn)
def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x), x)
        for x in client_ids
    ]
preprocessed_example_dataset = preprocess(example_dataset, "0")

sample_clients = client_ids

federated_train_data = make_federated_data(emnist_train, sample_clients)

#print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
#print('First dataset: {d}'.format(d=federated_train_data[0]))
#print('Element spec', preprocessed_example_dataset.element_spec)
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros', use_bias=False),
        tf.keras.layers.Softmax(),
    ])
def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#print(next(iter(preprocessed_example_dataset)))
#print(model_fn().forward_pass(next(iter(preprocessed_example_dataset))))
'''
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

str(iterative_process.initialize.type_signature)
state = iterative_process.initialize()
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))
NUM_ROUNDS = 11
for round_num in range(2, NUM_ROUNDS):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))
'''
