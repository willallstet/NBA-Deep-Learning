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
        probs_1 = self.feed_forward_1(whole_sequence_output)
        probs_2 = self.feed_forward_2(probs_1)
        return probs_2, (final_memory_state, final_carry_state)

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, probs))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):   
    for i in range(0, len(train_inputs)):
        print("Training Batch: ", i)
        game_data = train_inputs[i]
        # print(game_data)
        game_input = np.array(game_data[4] + game_data[5])
        if i == 9:
            print(game_data[3])
            print(game_data[2])
            print(game_data[0])
        game_input = np.reshape(game_input, (1, len(game_input), 1))
        game_result = train_labels[(game_data[3], game_data[2], game_data[0])]
        with tf.GradientTape() as tape:
            #print(game_input)
            results = model(game_input)
            loss = model.loss(results[0], game_result)
            #print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    probs = model.call(test_inputs)
    accuracy = model.accuracy(probs, test_labels)

def main():
    
    # TO-DO:  Separate your train and test data into inputs and labels
    train_x = preprocess.preprocess_games("archive/train_games.csv")
    # print("TEMPORARY -----------------------")
    # for i in range(10):
    #     print(train_x[i])
    test_x = preprocess.preprocess_games("archive/test_games.csv")
    odds_2021_dict = preprocess.preprocess_odds("archive/nba odds 2021-22.xlsx")
    odds_2020_dict = preprocess.preprocess_odds("archive/nba odds 2020-21.xlsx")
    odds_2019_dict = preprocess.preprocess_odds("archive/nba odds 2019-20.xlsx")
    odds_2018_dict = preprocess.preprocess_odds("archive/nba odds 2018-19.xlsx")
    odds_2017_dict = preprocess.preprocess_odds("archive/nba odds 2017-18.xlsx")
    odds_2016_dict = preprocess.preprocess_odds("archive/nba odds 2016-17.xlsx")
    train_y = {**odds_2019_dict, **odds_2018_dict, **odds_2017_dict, **odds_2016_dict}
    test_y = {**odds_2021_dict, **odds_2020_dict}

    # TODO: initialize model
    model = Model()

    # TODO: Set-up the training step
    train(model, train_x, train_y)

    acc = test(model, test_x, test_y)
    print("ACCURACY: ", acc)
    
if __name__ == '__main__':
    main()
