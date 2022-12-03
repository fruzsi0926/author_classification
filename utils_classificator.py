import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import copy

def inv_cat(input):
    return np.argmax(input)

def get_salient_map(data_batch, model, idx):
    #SALIENT MAP VISUALIZATION
    for train_gt in data_batch:
        break

    # input parameters
    input_img = train_gt[0]
    input_annotation = train_gt[1]
    index_in_batch = idx


    input_i = tf.convert_to_tensor(np.expand_dims(np.array(input_img)[index_in_batch], axis=0))
    input_a = input_annotation[index_in_batch]

    result = model(input_i)


    with tf.GradientTape() as tape:
        tape.watch(input_i)
        result = model(input_i)
        print(result)
        max_score = result[0,input_a]
    grads = np.array(np.squeeze(tape.gradient(max_score, input_i)))

    plot_grads = grads
    plot_img = np.array(input_img)[index_in_batch].astype("uint8")

    #sum R,G,B channels
    if len(plot_grads.shape) >2 and plot_grads.shape[2]==3:
        plot_2D= plot_grads[:,:,0]

    high_indices = plot_2D<np.mean(plot_2D)
    high_indices = np.stack((high_indices,high_indices,high_indices),axis=2)

    #cut the lower part of activations below the mean.
    print(np.mean(plot_grads))
    plot_grads = plot_grads*(255/np.max(plot_grads))


    #input image and  salient map
    #plt.imshow(plot_grads)
    #plt.show()
    #plt.imshow(plot_img)
    #plt.show()


    # input image + salient map (A version)
    plot_grads_filtered_A = copy.deepcopy(plot_grads)
    plot_grads_filtered_A[plot_grads_filtered_A<40]=0.0
    plt.imshow(plot_img, alpha=1.0)
    plt.imshow(plot_grads_filtered_A, alpha=0.3)
    plt.show()


    #input image + salient map (B version)
    #plot_grads[high_indices] = plot_img[high_indices]
    #plt.imshow(plot_img, alpha=1.0)
    #plt.imshow(plot_grads, alpha=0.7)
    #plt.show()

    return plot_img, plot_grads_filtered_A

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2):
    sum = (train_split + val_split)
    assert sum > 0.99 and sum < 1.01

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)   
    val_ds = ds.skip(train_size)

    return train_ds, val_ds