import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def train_step(gen,disc, imgs, loss_function, gen_optim,disc_optim, is_training):
    """
    Performs a forward and backward pass for one datapoint of our training set
      Args:
        - gen <tensorflow.keras.Model>: our created generator
        - disc <tensorflow.keras.Model>: our created disciminator
        - imgs <tensorflow.tensor>: our real images
        - loss_funcion <keras function>: function we used for calculating our losses
        - gen_optim <keras function>: our optimizer for the generator
        - disc_optim <keras function>: our optimizer for the disciminator
        - is_training <bool>: activates training functions for different layers of the gen and disc

      Returns:
        - loss <float>: our calculated loss for the datapoint
      """
    # generates a random image
    noise = tf.random.normal([64,100])

    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:

        generated_imgs = gen(noise,training= is_training)

        real_out = disc(imgs,training=is_training)
        fake_out = disc(generated_imgs,training= is_training)

        # calculating losses
        gen_loss = loss_function(tf.ones_like(fake_out), fake_out)
        disc_loss = loss_function(tf.ones_like(real_out)*0.9, real_out) + loss_function(tf.zeros_like(fake_out), fake_out)

        # calculaing the gradients
        gradients_gen = gen_tape.gradient(gen_loss, gen.trainable_variables)
        gradients_disc = disc_tape.gradient(disc_loss, disc.trainable_variables)

    # updating weights and biases
    gen_optim.apply_gradients(zip(gradients_gen, gen.trainable_variables))
    disc_optim.apply_gradients(zip(gradients_disc, disc.trainable_variables))

    return gen_loss, disc_loss


def test(gen,disc, test_data, loss_function, is_training):
    """
    Test our MLP, by going through our testing dataset,
    performing a forward pass and calculating loss and accuracy
      Args:
        - gen <tensorflow.keras.Model>: our created generator
        - disc <tensorflow.keras.Model>: our created disciminator
        - imgs <tensorflow.tensor>: our real images
        - loss_funcion <keras function>: function we used for calculating our losses
        - gen_optim <keras function>: our optimizer for the generator
        - disc_optim <keras function>: our optimizer for the disciminator
        - is_training <bool>: activates training functions for different layers of the gen and disc

      Returns:
          - loss <float>: our mean loss for this epoch
          - accuracy <float>: our mean accuracy for this epoch
    """

    # initializing lists for accuracys and loss
    accuracy_aggregator = []
    gen_loss_aggregator = []
    disc_loss_aggregator = []

    for imgs in test_data:
        # forward step
        noise = tf.random.normal([64,100])
        generated_imgs = gen(noise,training= is_training)
        real_out = disc(imgs,training=is_training)
        fake_out = disc(generated_imgs,training= is_training)

        # calculating loss
        gen_loss = loss_function(tf.ones_like(fake_out), fake_out)
        disc_loss = loss_function(tf.ones_like(real_out)*0.9, real_out) + loss_function(tf.zeros_like(fake_out)*0.95, fake_out)

        # add loss and accuracy to the lists
        gen_loss_aggregator.append(gen_loss.numpy())
        disc_loss_aggregator.append(disc_loss.numpy())


    # calculate the mean of the loss and accuracy (for this epoch)
    gen_loss = tf.reduce_mean(gen_loss_aggregator)
    disc_loss = tf.reduce_mean(disc_loss_aggregator)

    return gen_loss,disc_loss # accuracy


def visualize_stat(train_g_l,valid_g_l,train_d_l,valid_d_l):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
    Args:

    """

    fig, axs = plt.subplots(2, 1)
    #fig.set_size_inches(13, 6)
    # making a grid with subplots
    for j in range(1):
        axs[0].plot(train_g_l[j])
        axs[0].plot(valid_g_l[j])
        axs[1].plot(train_d_l[j])
        axs[1].plot(valid_d_l[j])
        axs[1].sharex(axs[0])

    fig.legend([" train_g_l"," valid_g_l"," train_d_l"," valid_d_l"],loc="lower right")
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()
