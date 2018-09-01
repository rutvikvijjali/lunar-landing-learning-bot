import gym
import tensorflow as tf
import numpy as np


def init_wts(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=1))


def init_bias(shape):
    return tf.Variable(tf.constant(0.1),shape)


def make_layer(input_layer,out_size):

    inp_size = int(input_layer.get_shape()[1])

    print('Inputsize:' ,inp_size)

    W = init_wts([inp_size,out_size])
    b = init_bias([out_size])
    return tf.matmul(input_layer,W)+b

def discount_rewards(rewards):

    discounted_episode_rewards = np.zeros_like(rewards)
    cumulative = 0
    for t in reversed(range(len(rewards))):
        cumulative = cumulative * 0.99 + rewards[t]
        discounted_episode_rewards[t] = cumulative

    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return discounted_episode_rewards


    # W*x + B
num_inputs = 8
num_hidden = 10
num_out = 4

inp_layer = tf.placeholder(shape=[None,num_inputs],dtype=tf.float32)
out_layer = tf.placeholder(shape=[None,num_out],dtype=tf.float32)
discount_reward = tf.placeholder(dtype=tf.float32)


log_probabilities = tf.placeholder(shape=[None,1],dtype=tf.float32)

h1 = tf.nn.relu(make_layer(inp_layer,num_hidden))

print('Layer 1 Made')

h2 = tf.nn.relu(make_layer(h1,num_hidden))

print('Layer 2 Made')

y_val = make_layer(h2,num_out)
y_prob = tf.nn.softmax(y_val)




# y_val_t = tf.transpose(y_val)

output = tf.nn.softmax(y_val)

out_layer_t = tf.transpose(out_layer)
#
# y_fin = tf.transpose(y_prob)

action = tf.argmax(y_prob[0])

log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=out_layer,logits=y_val)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
loss = tf.reduce_mean(discount_reward*log_prob)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

gamma = 0.95

rewards = []

env = gym.make('LunarLander-v2')

with tf.Session() as sess:
    sess.run(init)
    #saver.restore(sess, './models/REINFORCE-Lunar-Lander')
    for episode in range(2000):

        print('Episode Number: ',episode)
        pos = env.reset()

        steps = 0

        rewards = []

        gradient_placeholders = []

        observations = []

        sampled_actions = []

        true_actions = []

        log_probs = 0


        while(True):


            if(episode%100 == 0):
		
               	env.render()
		saver.save(sess, './models/REINFORCE-Lunar-Lander')


            observations.append(pos)




            y_ = output.eval(feed_dict={inp_layer: pos.reshape(1,8)})
            # print('Probability Distribution ',y_)
            # y = action.eval(feed_dict={inp_layer: pos.reshape(1,8)})

            act = np.random.choice(range(len(y_.ravel())), p = y_.ravel())
            pos, reward, done, info = env.step(act)
            rewards.append(reward)
            sampled = np.zeros(shape=[1, 4])
            sampled[0][act] = 1

            sampled_actions.append(sampled.reshape((1,4)))


            true_actions.append(y_)
            # sampled = sampled[0]
            # y_ = y_[0]-

            # print('Sampled probs, ', sampled)
            # print('val equals {}'.format(val[0][0]))



            # log_prob.eval(feed_dict={out_layer: sampled.reshape(1,4), y_prob : y_})



            # print(log_probs)

            # print('Position : {} , Action : {} , Reward : {}'.format(pos,y,reward))


            if done:

                s = np.shape(np.array(sampled_actions))[0]


                print('Reward :',sum(rewards))

                # prob_list = log_prob.eval(feed_dict={out_layer:np.array(sampled_actions).reshape(s,4),y_prob:np.array(true_actions).reshape(s,4)})


                # print(prob_list)


                print('Done at {}'.format(steps))


                disc_rew = discount_rewards(rewards)



                l = loss.eval(feed_dict={inp_layer:np.vstack(observations),out_layer:np.vstack(sampled_actions),discount_reward:disc_rew})

                sess.run(train,feed_dict={inp_layer:np.vstack(observations),out_layer:np.vstack(sampled_actions),discount_reward:disc_rew})

                print('Loss = ',l)


                #
                # sess.run(train,feed_dict={})
                #
                # print('Discounted Reward :',fin_grad)
                #
                #
                # fin = key_val.eval()
                # print('Final Gradient to minimize : ', fin )


                break
            steps += 1

            # sampled = sampled[0]
            # y_ = y_[0]-

            # print('Sampled probs, ', sampled)
            # print('val equals {}'.format(val[0][0]))
    # meta_graph_def = tf.train.export_meta_graph(filename='/models/REINFORCE-Lunar-Lander.meta')



print('Running the trained session')


with tf.Session() as sess:
    # https://www.tensorflow.org/api_guides/python/meta_graph
    # new_saver = tf.train.import_meta_graph('/models/REINFORCE-Lunar-Lander.meta')
    saver.restore(sess,'./models/REINFORCE-Lunar-Lander')

    for _ in range(100):
        pos = env.reset()
        reward_test = []
        
        
        
        while(True):
            env.render()
    
            y_ = sess.run(output,feed_dict={inp_layer: pos.reshape(1, 8)})
    
            act = np.random.choice(range(len(y_.ravel())), p=y_.ravel())
    
            pos, reward, done, info = env.step(act)
    
    
            reward_test.append(reward)
    
    
    
            if (done):
                print('Total Reward = ',sum(reward_test))
    
                break





