"""
In this task we are going to use the main.py given in the lecture and use it to print out emojies based on the words
I wanted to expand the task to take every character in the alphabet to create the emojies. Next can be to expand the list of emojis. 
"""
import torch
import torch.nn as nn
import string
import sys


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, label_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(char_encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_encoding_size)

    # Batch size is the number of sequences to be passed to the LSTM.
    # When training, batch size is typically > 1, but the batch size is 1 when generating
    def reset(self, batch_size=1):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, batch_size, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


if __name__ == '__main__':
    SEARCH_WORD = 'Simon'

    if len(sys.argv) < 2:  # If argument is not given
        print("Hint: Run $Python string_to_emoji.py \'<Search Word>\'")
    else:
        SEARCH_WORD = sys.argv[1]

    print("The word is {}".format(SEARCH_WORD))

    # Creates the char_enc with the whole alphabet uppercase and lowercase + " "
    alphabet = string.ascii_letters + ' '
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    index_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # one hot encode
    char_enc = list()
    for value in index_to_char:
        letter = [0. for _ in range(len(alphabet))]
        letter[value] = 1.
        char_enc.append(letter)
    char_encoding_size = len(char_enc)

    # Output
    emojis = {
        'hat': '\U0001F3A9',
        'cat': '\U0001F408',
        'rat': '\U0001F400',
        'flat': '\U0001F3E2',
        'matt': '\U0001F468',
        'cap': '\U0001F9E2',
        'son': '\U0001F466'
    }

    emoji_enc = [[1., 0., 0., 0., 0., 0., 0.],  # 'hat' 0
                 [0., 1., 0., 0., 0., 0., 0.],  # 'rat' 1
                 [0., 0., 1., 0., 0., 0., 0.],  # 'cat' 2
                 [0., 0., 0., 1., 0., 0., 0.],  # 'flat' 3
                 [0., 0., 0., 0., 1., 0., 0.],  # 'matt' 4
                 [0., 0., 0., 0., 0., 1., 0.],  # 'cap' 5
                 [0., 0., 0., 0., 0., 0., 1.]]  # 'son' 6
    emoji_encoding_size = len(emoji_enc)
    index_to_emoji = [emojis['hat'], emojis['rat'], emojis['cat'],
                      emojis['flat'], emojis['matt'], emojis['cap'], emojis['son']]

    x_train = torch.tensor([
        [[char_enc[alphabet.index('h')]], [char_enc[alphabet.index('a')]], [
            char_enc[alphabet.index('t')]], [char_enc[alphabet.index(' ')]]],
        [[char_enc[alphabet.index('r')]], [char_enc[alphabet.index('a')]], [
            char_enc[alphabet.index('t')]], [char_enc[alphabet.index(' ')]]],
        [[char_enc[alphabet.index('c')]], [char_enc[alphabet.index('a')]], [
            char_enc[alphabet.index('t')]], [char_enc[alphabet.index(' ')]]],
        [[char_enc[alphabet.index('f')]], [char_enc[alphabet.index('l')]], [
            char_enc[alphabet.index('a')]], [char_enc[alphabet.index('t')]]],
        [[char_enc[alphabet.index('m')]], [char_enc[alphabet.index('a')]], [
            char_enc[alphabet.index('t')]], [char_enc[alphabet.index('t')]]],
        [[char_enc[alphabet.index('c')]], [char_enc[alphabet.index('a')]], [
            char_enc[alphabet.index('p')]], [char_enc[alphabet.index(' ')]]],
        [[char_enc[alphabet.index('s')]], [char_enc[alphabet.index('o')]], [char_enc[alphabet.index('n')]], [char_enc[alphabet.index(' ')]]]])  # 'hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son '

    y_train = torch.tensor([
        [emoji_enc[0], emoji_enc[0], emoji_enc[0], emoji_enc[0]],
        [emoji_enc[1], emoji_enc[1], emoji_enc[1], emoji_enc[1]],
        [emoji_enc[2], emoji_enc[2], emoji_enc[2], emoji_enc[2]],
        [emoji_enc[3], emoji_enc[3], emoji_enc[3], emoji_enc[3]],
        [emoji_enc[4], emoji_enc[4], emoji_enc[4], emoji_enc[4]],
        [emoji_enc[5], emoji_enc[5], emoji_enc[5], emoji_enc[5]],
        [emoji_enc[6], emoji_enc[6], emoji_enc[6], emoji_enc[6]]])

    model = LongShortTermMemoryModel(char_encoding_size, emoji_encoding_size)

    def generate(string):
        model.reset()
        for i in range(len(string)):
            char_index = list(index_to_char.keys())[list(
                index_to_char.values()).index(string[i])]
            y = model.f(torch.tensor([[char_enc[char_index]]]))
            if i == len(string) - 1:
                print(index_to_emoji[y.argmax(1)])

    optimizer = torch.optim.RMSprop(model.parameters(), 0.001)  # 0.001
    for epoch in range(500):
        for i in range(x_train.size()[0]):
            model.reset()
            model.loss(x_train[i], y_train[i]).backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 9:
            generate(SEARCH_WORD)
