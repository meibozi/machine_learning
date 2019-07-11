import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
from time import time

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)

        ax.imshow(np.reshape(images[idx],(28,28)))
        title = "label="+str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])

        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx += 1
    plt.show()

def layer(output_dim,input_dim,inputs,activation=None):
    W = tf.Variable(tf.random_normal([input_dim,output_dim]))
    b = tf.Variable(tf.random_normal([1,output_dim]))
    XWb = tf.matmul(inputs,W)+b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


def layer_debug(output_dim,input_dim,inputs,activation=None):
    W = tf.Variable(tf.random_normal([input_dim,output_dim]))
    b = tf.Variable(tf.random_normal([1,output_dim]))
    XWb = tf.matmul(inputs,W)+b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs,W,b

def plot_image(image):
    plt.imshow(image.reshape(28,28))
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('train',mnist.train.num_examples,
      ',validation',mnist.validation.num_examples,
      ',test',mnist.test.num_examples)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.train.labels[0])
print(np.argmax(mnist.train.labels[0]))
#plot_images_labels_prediction(mnist.train.images,mnist.train.labels,[],0)






x = tf.placeholder("float",[None,784])
h1 = layer(output_dim=1000,input_dim=784,inputs=x,activation=tf.nn.relu)
h2 = layer(output_dim=1000,input_dim=1000,inputs=h1,activation=tf.nn.relu)
y_predict = layer(output_dim=10,input_dim=1000,inputs=h2,activation=None)
y_label = tf.placeholder("float",[None,10])

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(y_label,1),tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
loss_list = [];epoch_list = []; accuracy_list = []
startTime = time()





with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(trainEpochs):
        for i in range(totalBatchs):
            batch_x,batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer,feed_dict={x:batch_x,y_label:batch_y})

        loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y_label:mnist.validation.labels})

        epoch_list.append(epoch_list)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print("Train Epoch:",'%02d'%(epoch+1),"Loss=","{:.9f}".format(loss),"Accuracy=",acc)

    duration = time()-startTime
    print("Train Finished takes:",duration)

    print("Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_label:mnist.test.labels}))




    tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log/area/', sess.graph)
    prediction_result = sess.run(tf.argmax(y_predict,1),feed_dict={x:mnist.test.images})





print(prediction_result[:10])
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,0)