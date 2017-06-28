# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import matplotlib.pyplot as plt

def ReconsAE(predictions, sess):
    # Plot reconstructions for the basic autoencoder
    f_recons = plt.figure(1)
    f_recons.suptitle('Reconstructions: originals, reconstructions')
    for p in range(0, 4):
        plt.subplot(4, 2, 2 * p + 1)
        temp1 = sess.run(predictions[0])
        temp1 = temp1[p, :, 12, :, 0]
        temp1.reshape(24, 24)
        plt.imshow(temp1, cmap='gray')
        plt.subplot(4, 2, 2 * p + 2)
        temp2 = sess.run(predictions[1])
        temp2 = temp2[p, :, 12, :, 0]
        temp2.reshape(24, 24)
        plt.imshow(temp2, cmap='gray')
    plt.pause(0.0001)

def ReconsVAE(predictions, sess):
    # Plot reconstructions for the basic variational autoencoder
    f_recons = plt.figure(1)
    f_recons.suptitle('Reconstructions: originals, predicted means, predicted variances')
    for p in range(0, 4):
        plt.subplot(4, 3, 3 * p + 1)
        plt.xticks([])
        plt.yticks([])
        temp1 = sess.run(predictions[4])
        temp1 = temp1[p, :, 12, :, 0]
        temp1.reshape(24, 24)
        plt.imshow(temp1, cmap='gray')
        plt.subplot(4, 3, 3 * p + 2)
        plt.xticks([])
        plt.yticks([])
        temp2 = sess.run(predictions[2])
        temp2 = temp2[p, :, 12, :, 0]
        temp2.reshape(24, 24)
        plt.imshow(temp2, cmap='gray')
        plt.subplot(4, 3, 3 * p + 3)
        plt.xticks([])
        plt.yticks([])
        temp3 = sess.run(predictions[5])
        temp3 = temp3[p, :, 12, :, 0]
        temp3.reshape(24, 24)
        plt.imshow(temp3, cmap='gray')
    plt.pause(0.0001)