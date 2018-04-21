import os, inspect, time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(sess, neuralnet, saver, dataset, iteration):

    start_time = time.time()
    loss_tr = 0
    list_loss = []

    try: os.mkdir(PACK_PATH+"/training")
    except: pass

    print("\nTraining SRCNN to %d iterations" %(iteration))
    train_writer = tf.summary.FileWriter(PACK_PATH+'/logs')
    for it in range(iteration):

        X_tr, Y_tr = dataset.next_batch()
        summaries, _ = sess.run([neuralnet.summaries, neuralnet.optimizer], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
        loss_tr = sess.run(neuralnet.loss, feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
        list_loss.append(loss_tr)
        train_writer.add_summary(summaries, it)

        if(it % 100 == 0):
            img_recon = sess.run(neuralnet.recon, feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            img_recon = np.squeeze(img_recon, axis=0)
            img_ground = np.squeeze(Y_tr, axis=0)

            img_recon = img_recon / (np.max(img_recon) - np.min(img_recon))
            img_recon += abs(np.min(img_recon))

            plt.clf()
            plt.subplot(121)
            plt.title("Prediction")
            plt.imshow(img_recon)
            plt.subplot(122)
            plt.title("Ground-Truth")
            plt.imshow(img_ground)
            plt.savefig("%s/training/%d.png" %(PACK_PATH, it))

        print("Iteration [%d / %d] | Loss: %f" %(it, iteration, loss_tr))

        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("Final iteration | Loss: %f" %(loss_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    list_loss = np.asarray(list_loss)
    np.save("loss", list_loss)

    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(list_loss, color='blue', linestyle="-", label="loss")
    # plt.plot(np.log(list_loss), color='tomato', linestyle="--", label="log scale loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    # plt.legend(loc='best')
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("loss.png")

    if(list_loss.shape[0] > 100):
        sparse_loss_x = np.zeros((100))
        sparse_loss = np.zeros((100))

        unit = int(list_loss.shape[0]/100)
        for i in range(100):
            sparse_loss_x[i] = i * unit
            sparse_loss[i] = list_loss[i * unit]

        plt.clf()
        plt.rcParams['font.size'] = 15
        plt.plot(sparse_loss_x, sparse_loss, color='blue', linestyle="-", label="loss")
        # plt.plot(sparse_loss_x, np.log(sparse_loss), color='tomato', linestyle="--", label="log scale loss")
        plt.ylabel("loss")
        plt.xlabel("iteration")
        # plt.legend(loc='best')
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.savefig("loss_sparse-ver.png")

# def validation(sess, neuralnet, saver,
#                dataset, sequence_length):
#
#     if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
#         saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")
#
#     start_time = time.time()
#     print("\nValidation")
#     for at in range(dataset.am_total - (sequence_length * 2)):
#
#         if(dataset.is_norm): addnorm = True
#         else: addnorm = False
#
#         X_val, Y_val = dataset.next_batch(batch_size=1, sequence_length=sequence_length, train=False)
#         l2dist = sess.run(neuralnet.loss, feed_dict={neuralnet.inputs:X_val, neuralnet.outputs:Y_val})
#
#         if(addnorm): list_norm.append(l2dist)
#         else: list_anom.append(l2dist)
#
#         if(at % 100 == 0):
#             print("Validation [%d / %d] | L2 Dist: %f" %(at, dataset.am_total, l2dist))
#
#     elapsed_time = time.time() - start_time
#     print("Elapsed: "+str(elapsed_time))
