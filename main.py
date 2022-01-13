from gan import Descriminator,Generator
from util import train_step, test, visualize_stat
import tensorflow as tf
from classify import classify
from create_data import load_data

tf.keras.backend.clear_session()
train_ds, valid_ds, test_ds = load_data()

disc = Descriminator()
gen = Generator()

train_gen_losses = []
valid_gen_losses = []
train_disc_losses = []
valid_disc_losses = []
#valid_accuracies = []

with tf.device('/device:gpu:0'):
# training the model
    results,trained_gen,trained_disc = classify(gen,disc, 10, train_ds, valid_ds)

    # saving results for visualization
    train_gen_losses.append(results[0])
    valid_gen_losses.append(results[1])
    train_disc_losses.append(results[2])
    valid_disc_losses.append(results[3])
    #valid_accuracies.append(results[2])

# testing the trained model
# (this code snippet should only be inserted when one decided on all hyperparameters)
#_, test_accuracy = test(trained_model, test_ds, tf.keras.losses.BinaryCrossentropy(), False)
#print("Accuracy (test set):", test_accuracy)

# visualizing losses and accuracy
visualize_stat(train_gen_losses, valid_gen_losses, train_disc_losses,valid_disc_losses)
