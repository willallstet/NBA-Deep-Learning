import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import preprocess
import matplotlib.pyplot as plt

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.rnn_size = 10
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.feed_forward_1 = tf.keras.layers.Dense(16, activation='relu')
        self.feed_forward_2 = tf.keras.layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        whole_sequence_output, final_memory_state, final_carry_state = self.lstm(inputs)
        probs_1 = self.feed_forward_1(whole_sequence_output)
        probs_2 = self.feed_forward_2(probs_1)
        return probs_2, (final_memory_state, final_carry_state)

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, probs))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    loss_list = []
    for i in range(0, len(train_inputs),3):
        print("TRAINING GAME #", i)
        game_data = [train_inputs[i],train_inputs[i+1],train_inputs[i+2]]
        game_input = np.array([game_data[0][4] + game_data[0][5],game_data[1][4] + game_data[1][5],game_data[2][4] + game_data[2][5]])
        game_input = np.reshape(game_input, (1, len(game_input), 12))
        game_result = [train_labels[(game_data[0][3], game_data[0][2], game_data[0][0])],train_labels[(game_data[1][3], game_data[1][2], game_data[1][0])],train_labels[(game_data[2][3], game_data[2][2], game_data[2][0])]]
        with tf.GradientTape() as tape:
            results, _ = model(game_input)
            results = tf.squeeze(results)
            loss = model.loss(results, tf.one_hot(game_result,2))
            loss_list.append(loss)
        if i == 0:
            print(loss)
        if i == len(train_inputs)-1:
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    plt.plot(range(int(len(train_inputs) / 3)), loss_list)
    plt.show()

def test(model, test_inputs, test_labels):
    correct_total = 0
    total = 0
    for i in range(0, len(test_inputs),3):
        print("TEST GAME #", i)
        if((i+2)<(len(test_inputs)-1)):
            total+=3
            game_data = [test_inputs[i],test_inputs[i+1],test_inputs[i+2]]
            game_input = game_input = np.array([game_data[0][4] + game_data[0][5],game_data[1][4] + game_data[1][5],game_data[2][4] + game_data[2][5]])
            game_input = np.reshape(game_input, (1, len(game_input), 12))
            test_result = [test_labels[(game_data[0][3], game_data[0][2], game_data[0][0])],test_labels[(game_data[1][3], game_data[1][2], game_data[1][0])],test_labels[(game_data[2][3], game_data[2][2], game_data[2][0])]]
            results, _ = model(game_input)
            #print(results)
            #score = tf.reduce_sum(results)
            #if score < 0.5:
            #    score = 0
            #elif score > 0.5:
            #    score = 18
            results=tf.squeeze(results)
            if tf.math.argmax(results[0]) == test_result[0]:
                correct_total += 1
            if tf.math.argmax(results[1]) == test_result[1]:
                correct_total += 1
            if tf.math.argmax(results[2]) == test_result[2]:
                correct_total += 1
    return correct_total / total

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

    print("FINISHES TRAIN ----------------------------------------")
    acc = test(model, test_x, test_y)
    print("ACCURACY: ", acc)
    
if __name__ == '__main__':
    main()
