"""
In this task we are going to use the main.py given in the lecture and use it to print out emojies based on the words
"""
import torch
import torch.nn as nn
import string
import sys


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, label_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(char_encoding_size, 128)  # 128 is the state size
        self.fc1 = nn.Linear(128, emoji_encoding_size)

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
        return self.fc1(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


if __name__ == '__main__':
    SEARCH_WORD = ' rt '

    # Tried to make it possible to run with search word in command, got weird import error
    """ 
    if len(sys.argv) < 2:  # If argument is not given
        print("Hint: Run $Python string_to_emoji.py \'<Search Word>\'")
    else:
        SEARCH_WORD = sys.argv[1]
    """
    print("The word is {}:".format(SEARCH_WORD))

    # Input in text form
    char_enc = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 2
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 3
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' 4
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 5
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'f' 6
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'l' 7
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'm' 8
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p' 9
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 's' 10
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'o' 11
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # 'n' 12
    ]
    char_encoding_size = len(char_enc)
    index_to_char = [' ', 'h', 'a', 't', 'r',
                     'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

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

    emoji_enc = [
        [1., 0., 0., 0., 0., 0., 0.],  # 'hat' 0
        [0., 1., 0., 0., 0., 0., 0.],  # 'rat' 1
        [0., 0., 1., 0., 0., 0., 0.],  # 'cat' 2
        [0., 0., 0., 1., 0., 0., 0.],  # 'flat' 3
        [0., 0., 0., 0., 1., 0., 0.],  # 'matt' 4
        [0., 0., 0., 0., 0., 1., 0.],  # 'cap' 5
        [0., 0., 0., 0., 0., 0., 1.]   # 'son' 6
    ]
    emoji_encoding_size = len(emoji_enc)
    index_to_emoji = [emojis['hat'], emojis['rat'], emojis['cat'],
                      emojis['flat'], emojis['matt'], emojis['cap'], emojis['son']]

    x_train = torch.tensor([
        [[char_enc[1]], [char_enc[2]], [char_enc[3]], [char_enc[0]]],
        [[char_enc[4]], [char_enc[2]], [char_enc[3]], [char_enc[0]]],
        [[char_enc[5]], [char_enc[2]], [char_enc[3]], [char_enc[0]]],
        [[char_enc[6]], [char_enc[7]], [char_enc[2]], [char_enc[3]]],
        [[char_enc[8]], [char_enc[2]], [char_enc[3]], [char_enc[3]]],
        [[char_enc[5]], [char_enc[2]], [char_enc[9]], [char_enc[0]]],
        [[char_enc[10]], [char_enc[11]], [char_enc[12]], [char_enc[0]]]])  # 'hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son '

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
            char_index = index_to_char.index(string[i])
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
