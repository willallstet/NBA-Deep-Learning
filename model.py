import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import preprocess

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.rnn_size = 10
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.feed_forward_1 = tf.keras.layers.Dense(32, activation='relu')
        self.feed_forward_2 = tf.keras.layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        whole_sequence_output, final_memory_state, final_carry_state = self.lstm(inputs)
        probs = self.feed_forward(whole_sequence_output)
        return probs, (final_memory_state, final_carry_state)

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, probs))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):   
    for i in range(0, len(train_inputs)):
        print("Training Batch: ", i)
        game_input = train_inputs[i]
        game_result = train_labels[i]
        with tf.GradientTape() as tape:
            results = model(game_input, None)
            loss = model.loss(results[0], game_result)
            #print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    probs = model.call(test_inputs)
    accuracy = model.accuracy(probs, test_labels)

def main():
    
    # TO-DO:  Separate your train and test data into inputs and labels
    years_dict = preprocess("archive/games.csv")

    # TODO: initialize model
    model = Model()

    # TODO: Set-up the training step
    train(model, train_x, train_y)

    acc = test(model, test_x, test_y)
    print("ACCURACY: ", acc)
    
if __name__ == '__main__':
    main()
