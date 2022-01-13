import argparse
from gan import Descriminator,Generator
from util import train_step, test, visualize_stat
import tensorflow as tf
import matplotlib.pyplot as plt


def classify(gen,disc, num_epochs, train_ds, valid_ds):
    """
    Trains and tests our predefined model.
        Args:
            - model <tensorflow.keras.Model>: our untrained model
            - optimizer <keras function>: optimizer for the model
            - num_epochs <int>: number of training epochs
            - train_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our training dataset
            - valid_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our validation set for testing and optimizing hyperparameters
        Returns:
            - results <list<list<float>>>: list with losses and accuracies
            - model <tensorflow.keras.Model>: our trained MLP model
    """
    seed = tf.random.normal([64,100])
    tf.keras.backend.clear_session()

    # initialize the loss: categorical cross entropy
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    lr = 0.0002

    gen_optim = tf.keras.optimizers.Adam(lr,beta_1=0.5)
    disc_optim = tf.keras.optimizers.Adam(lr,beta_1=0.5)

    # initialize lists for later visualization.
    train_gen_losses = []
    valid_gen_losses = []
    train_disc_losses = []
    valid_disc_losses = []
    #valid_accuracies = []

    # testing on our valid_ds once before we begin
    valid_gen_loss,valid_disc_loss = test(gen,disc, valid_ds, loss, is_training=False)
    valid_gen_losses.append(valid_gen_loss)
    valid_disc_losses.append(valid_disc_loss)
    #valid_accuracies.append(valid_accuracy)

    # Testing on our train_ds once before we begin
    train_gen_loss, train_disc_loss = test(gen,disc, train_ds, loss, is_training=False)
    train_gen_losses.append(train_gen_loss)
    train_disc_losses.append(train_disc_loss)

    # training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f' starting with (validation set): gen_loss {valid_gen_losses[-1]} and disc_loss {valid_disc_losses[-1]}')
        print(f' and (training set): gen_loss {train_gen_losses[-1]} and disc_loss {train_disc_losses[-1]}')
        print("{}/{} epoches".format(epoch, num_epochs))
        
        # training (and calculating loss while training)
        epoch_gen_loss_agg = []
        epoch_disc_loss_agg = []

        for imgs in train_ds:
            train_gen_loss,train_disc_loss = train_step(gen, disc, imgs, loss, gen_optim,disc_optim, is_training=True)
            epoch_gen_loss_agg.append(train_gen_loss)
            epoch_disc_loss_agg.append(train_disc_loss)

        # track training loss
        train_gen_losses.append(tf.reduce_mean(epoch_gen_loss_agg))
        train_disc_losses.append(tf.reduce_mean(epoch_disc_loss_agg))

        ## After ith epoch plot image
        if (epoch % 5) == 0:
            fake_image = tf.reshape(gen(seed, training=False), shape = (64,28,28))
            plt.imshow(fake_image[10], cmap = "gray")

            #plt.imsave("{}/{}.png".format(OUTPUT_DIR,epoch),fake_image, cmap = "gray")
            plt.show()

        # testing our model in each epoch to track accuracy and loss on the validation set
        valid_gen_loss, valid_disc_loss = test(gen,disc, valid_ds, loss,is_training= False)
        valid_gen_losses.append(valid_gen_loss)
        valid_disc_losses.append(valid_disc_loss)
        #valid_accuracies.append(valid_accuracy)

    results = [train_gen_losses, valid_gen_losses, train_disc_losses, valid_disc_losses]
    return results, gen, disc
