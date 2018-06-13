import os, inspect, time

import scipy.misc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(sess, neuralnet, saver, dataset, iteration, batch_size):

    start_time = time.time()
    loss_tr = 0
    list_loss = []

    try: os.mkdir(PACK_PATH+"/training")
    except: pass
    try: os.mkdir(PACK_PATH+"/static")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/bicubic")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/reconstruction")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/high-resolution")
    except: pass


    print("\nTraining SRCNN to %d iterations" %(iteration))
    train_writer = tf.summary.FileWriter(PACK_PATH+'/logs')
    for it in range(iteration):

        X_tr, Y_tr = dataset.next_batch()
        summaries, _ = sess.run([neuralnet.summaries, neuralnet.optimizer], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
        loss_tr, psnr_tr = sess.run([neuralnet.loss, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
        list_loss.append(loss_tr)
        # train_writer.add_summary(summaries, it)

        if(it % 100 == 0):
            np.save("loss", np.asarray(list_loss))

            randidx = int(np.random.randint(dataset.amount_tr, size=1))
            X_tr, Y_tr = dataset.next_batch(idx=randidx, train=True)

            img_recon = sess.run(neuralnet.recon, feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            img_input = np.squeeze(X_tr, axis=0)
            img_recon = np.squeeze(img_recon, axis=0)
            img_ground = np.squeeze(Y_tr, axis=0)

            plt.clf()
            plt.rcParams['font.size'] = 100
            plt.figure(figsize=(100, 40))
            plt.subplot(131)
            plt.title("Low-Resolution")
            plt.imshow(img_input, cmap='gray')
            plt.subplot(132)
            plt.title("Reconstruction")
            plt.imshow(img_recon, cmap='gray')
            plt.subplot(133)
            plt.title("High-Resolution")
            plt.imshow(img_ground, cmap='gray')
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.savefig("%s/training/%d.png" %(PACK_PATH, it))
            plt.close()

            """static img(test)"""
            X_tr, Y_tr = dataset.next_batch(batch_size=batch_size, idx=int(0))
            img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            img_recon = np.squeeze(img_recon, axis=0)
            scipy.misc.imsave("%s/static/reconstruction/%d_psnr_%.3f.png" %(PACK_PATH, it, tmp_psnr), img_recon)

            if(it == 0):
                img_input = np.squeeze(X_tr, axis=0)
                img_ground = np.squeeze(Y_tr, axis=0)
                scipy.misc.imsave("%s/static/bicubic/%d.png" %(PACK_PATH, it), img_input)
                scipy.misc.imsave("%s/static/high-resolution/%d.png" %(PACK_PATH, it), img_ground)

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

def validation(sess, neuralnet, saver, dataset):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    start_time = time.time()
    print("\nValidation")
    for tidx in range(dataset.amount_te):

        X_te, Y_te = dataset.next_batch(idx=int(tidx))
        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_te, neuralnet.outputs:Y_te})
        img_recon = np.squeeze(img_recon, axis=0)
        scipy.misc.imsave("%s/static/reconstruction/%d_psnr_%.3f.png" %(PACK_PATH, it, tmp_psnr), img_recon)

        img_input = np.squeeze(X_te, axis=0)
        img_ground = np.squeeze(Y_te, axis=0)
        scipy.misc.imsave("%s/static/bicubic/%d.png" %(PACK_PATH, it), img_input)
        scipy.misc.imsave("%s/static/high-resolution/%d.png" %(PACK_PATH, it), img_ground)

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))
