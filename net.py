import tensorflow as tf
import numpy as np
import math
import os
import datetime

# https://www.tensorflow.org/how_tos/summaries_and_tensorboard/
# tensorboard --logdir=logs/rocks2
# open localhost:6006

class Net(object):
    def __init__(self, dir_path=None, log_path=None, **kwargs):
        
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
        
        self.setup()
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        
        self.was_restored = False
        if dir_path:
            if dir_path[-1] != '/': dir_path += '/'
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            self.dir_path = dir_path
            
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.dir_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                self.was_restored = True
                print "Restored model from checkpoint {0}".format(ckpt.model_checkpoint_path)
        else:
            self.saver = None
        
        if log_path:
            self.train_writer = tf.train.SummaryWriter(log_path, self.session.graph)
        else:
            self.train_writer = None
        
    def setup(self):
        raise NotImplementedError("Subclasses should implement setup()")
        
    def train(self, inputs, outputs):
        raise NotImplementedError("Subclasses should implement train()")
    
    def predict(self, inputs):
        raise NotImplementedError("Subclasses should implement predict()")
    
    def save(self, step):
        if self.saver:
            self.saver.save(self.session, self.dir_path + 'model.ckpt', global_step=step)
    
    def evaluate(self, inputs, outputs):
        raise NotImplementedError("Subclasses should implement evaluate()")
    
    def training_loop(self, training_batch_generator, testing_batch_generator, evaluation_interval=10):
        step = 0
        for step, (inp, out) in enumerate(training_batch_generator):
            self.train(inp, out)
            if step % evaluation_interval == 0:
                inp, out = next(testing_batch_generator)
                print self.evaluate(inp, out)
                self.save(step)
        print self.evaluate(inp, out)
        self.save(step)

def random_batch(inputs, outputs, count=100):
    indices = np.random.randint(0, len(inputs)-1, count)
    return inputs.take(indices, axis=0), outputs.take(indices, axis=0)

def batch_generator(inputs, outputs, size=100, batches=None, epochs=None, random=False, print_progress=False):
    if epochs is not None:
        batches = int(math.ceil(len(inputs) * 1.0 / size))
    
    last_printed = datetime.datetime.now()
    
    step = 0
    while True:
        if random:
            indices = np.random.randint(0, len(inputs)-1, size)
            yield inputs.take(indices, axis=0), outputs.take(indices, axis=0)
        else:
            start_index = step * size % len(inputs)
            end_index = min(start_index + size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
        step += 1
        if print_progress and batches and (datetime.datetime.now() - last_printed).total_seconds() > 4:
            last_printed = datetime.datetime.now()
            print "{0}%".format(step * 100.0 / batches)
        if batches is not None and step >= batches:
            break

