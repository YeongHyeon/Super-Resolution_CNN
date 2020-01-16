import os, inspect, time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def makedir(path):
    try: os.mkdir(path)
    except: pass

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def training(sess, neuralnet, saver, dataset, epochs, batch_size):

    start_time = time.time()
    loss_tr = 0
    list_loss = []
    list_psnr = []
    list_psnr_static = []

    makedir(PACK_PATH+"/training")
    makedir(PACK_PATH+"/static")
    makedir(PACK_PATH+"/static/reconstruction")

    print("\nTraining SRCNN to %d epochs" %(epochs))
    train_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint')

    X_static, Y_static, _ = dataset.next_train(batch_size=1)
    img_input = np.squeeze(X_static, axis=0)
    img_ground = np.squeeze(Y_static, axis=0)
    plt.imsave("%s/static/bicubic.png" %(PACK_PATH), img_input)
    plt.imsave("%s/static/high-resolution.png" %(PACK_PATH), img_ground)

    iteration = 0
    for epoch in range(epochs):

        while(True):
            X_tr, Y_tr, terminator = dataset.next_train(batch_size=batch_size)
            summaries, _ = sess.run([neuralnet.summaries, neuralnet.optimizer], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            loss_tr, psnr_tr = sess.run([neuralnet.loss, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            list_loss.append(loss_tr)
            list_psnr.append(psnr_tr)
            train_writer.add_summary(summaries, iteration)
            iteration += 1

            if(terminator): break

        X_tmp, Y_tmp = np.expand_dims(X_tr[0], axis=0), np.expand_dims(Y_tr[0], axis=0)

        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tmp, neuralnet.outputs:Y_tmp})
        img_input, img_recon, img_ground = np.squeeze(X_tmp, axis=0), np.squeeze(img_recon, axis=0), np.squeeze(Y_tmp, axis=0)

        plt.clf()
        plt.rcParams['font.size'] = 100
        plt.figure(figsize=(100, 40))
        plt.subplot(131)
        plt.title("Low-Resolution")
        plt.imshow(img_input)
        plt.subplot(132)
        plt.title("Reconstruction")
        plt.imshow(img_recon)
        plt.subplot(133)
        plt.title("High-Resolution")
        plt.imshow(img_ground)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.savefig("%s/training/%09d_psnr_%d.png" %(PACK_PATH, epoch, int(tmp_psnr)))
        plt.close()

        """static img(test)"""
        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_static, neuralnet.outputs:Y_static})
        list_psnr_static.append(tmp_psnr)
        img_recon = np.squeeze(img_recon, axis=0)
        plt.imsave("%s/static/reconstruction/%09d_psnr_%d.png" %(PACK_PATH, epoch, int(tmp_psnr)), img_recon)

        print("Epoch [%d / %d] | Loss: %f  PSNR: %f" %(epoch, epochs, loss_tr, psnr_tr))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("Final Epcoh | Loss: %f  PSNR: %f" %(loss_tr, psnr_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_loss, xlabel="Iteration", ylabel="L2 loss", savename="loss")
    save_graph(contents=list_psnr, xlabel="Iteration", ylabel="PSNR (dB)", savename="psnr")
    save_graph(contents=list_psnr_static, xlabel="Iteration", ylabel="PSNR (dB)", savename="psnr_static")

def validation(sess, neuralnet, saver, dataset):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    makedir(PACK_PATH+"/test")
    makedir(PACK_PATH+"/test/reconstruction")

    start_time = time.time()
    print("\nValidation")
    for tidx in range(dataset.amount_te):

        X_te, Y_te = dataset.next_test()
        if(X_te is None): break

        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_te, neuralnet.outputs:Y_te})
        img_recon = np.squeeze(img_recon, axis=0)
        plt.imsave("%s/test/reconstruction/%09d_psnr_%d.png" %(PACK_PATH, tidx, int(tmp_psnr)), img_recon)

        img_input = np.squeeze(X_te, axis=0)
        img_ground = np.squeeze(Y_te, axis=0)
        plt.imsave("%s/test/bicubic.png" %(PACK_PATH), img_input)
        plt.imsave("%s/test/high-resolution.png" %(PACK_PATH), img_ground)

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))
