"""
A from scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import common_utils

EMBED_SIZE = 20
NUMBER_OF_TRANSFORMER_BLOCKS = 6
FORWARD_EXPANSION = 4
NUMBER_OF_ATTENTION_HEADS = 4


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class Classifier(nn.Module):
    def __init__(self, sequence_len,
                 src_vocab_size,
                 src_pad_idx,
                 embed_size=EMBED_SIZE,
                 num_layers=NUMBER_OF_TRANSFORMER_BLOCKS,
                 forward_expansion=4,
                 heads=NUMBER_OF_ATTENTION_HEADS,
                 dropout=0,
                 device="cpu",
                 max_length=100):
        super(Classifier, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_len * embed_size, 1),
            nn.Sigmoid(),
        )

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, input_seq):
        src_mask = self.make_src_mask(input_seq)
        enc_src = self.encoder(input_seq, src_mask)
        output_probability = self.feed_forward(enc_src)
        return output_probability


# class DecoderBlock(nn.Module):
#     def __init__(self, embed_size, heads, forward_expansion, dropout, device):
#         super(DecoderBlock, self).__init__()
#         self.norm = nn.LayerNorm(embed_size)
#         self.attention = SelfAttention(embed_size, heads=heads)
#         self.transformer_block = TransformerBlock(
#             embed_size, heads, dropout, forward_expansion
#         )
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, value, key, src_mask, trg_mask):
#         attention = self.attention(x, x, x, trg_mask)
#         query = self.dropout(self.norm(attention + x))
#         out = self.transformer_block(value, key, query, src_mask)
#         return out
#
#
# class Decoder(nn.Module):
#     def __init__(
#         self,
#         trg_vocab_size,
#         embed_size,
#         num_layers,
#         heads,
#         forward_expansion,
#         dropout,
#         device,
#         max_length,
#     ):
#         super(Decoder, self).__init__()
#         self.device = device
#         self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
#         self.position_embedding = nn.Embedding(max_length, embed_size)
#
#         self.layers = nn.ModuleList(
#             [
#                 DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.fc_out = nn.Linear(embed_size, trg_vocab_size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, enc_out, src_mask, trg_mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
#
#         for layer in self.layers:
#             x = layer(x, enc_out, enc_out, src_mask, trg_mask)
#
#         out = self.fc_out(x)
#
#         return out


# class Transformer(nn.Module):
#     def __init__(
#             self,
#             src_vocab_size,
#             trg_vocab_size,
#             src_pad_idx,
#             trg_pad_idx,
#             embed_size=512,
#             num_layers=6,
#             forward_expansion=4,
#             heads=8,
#             dropout=0,
#             device="cpu",
#             max_length=100,
#     ):
#         super(Transformer, self).__init__()
#
#         self.encoder = Encoder(
#             src_vocab_size,
#             embed_size,
#             num_layers,
#             heads,
#             device,
#             forward_expansion,
#             dropout,
#             max_length,
#         )
#
#         # self.decoder = Decoder(
#         #     trg_vocab_size,
#         #     embed_size,
#         #     num_layers,
#         #     heads,
#         #     forward_expansion,
#         #     dropout,
#         #     device,
#         #     max_length,
#         # )
#
#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx
#         self.device = device
#
#     def make_src_mask(self, src):
#         src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
#         # (N, 1, 1, src_len)
#         return src_mask.to(self.device)
#
#     def make_trg_mask(self, trg):
#         N, trg_len = trg.shape
#         trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
#             N, 1, trg_len, trg_len
#         )
#
#         return trg_mask.to(self.device)
#
#     def forward(self, src, trg):
#         src_mask = self.make_src_mask(src)
#         trg_mask = self.make_trg_mask(trg)
#         enc_src = self.encoder(src, src_mask)
#         # out = self.decoder(trg, enc_src, src_mask, trg_mask)
#         return enc_src


def train_model_single_batch(model, epochs, input_sequences, results, optim, print_every=100):
    model.train()

    total_loss = 0

    for epoch in range(epochs):
        # for i, batch in enumerate(input_sequences):
        #     results = all_answers[i]
        #     src = batch.English.transpose(0, 1)
        #     trg = batch.French.transpose(0, 1)
        # the French sentence we input has all words except
        # the last, as it is using each word to predict the next

        # trg_input = trg[:, :-1]

        # the words we are trying to predict

        # create function to make masks using mask code above

        # src_mask, trg_mask = create_masks(src, trg_input)

        preds = model(input_sequences)
        # , trg_input, src_mask, trg_mask)

        optim.zero_grad()

        loss = F.binary_cross_entropy(preds, results.view(-1, preds.size(-1))
                                      # , ignore_index=target_pad
                                      )
        loss.backward()
        optim.step()

        # total_loss += loss.data
        # # if (i + 1) % print_every == 0:
        # loss_avg = total_loss / print_every
        print(f'loss at {epoch} epoch is: {loss.data}')


def get_acc_measure(predictions, actuals, size):
    print(f'shape of preds: {predictions.shape} and shape of actuals: {actuals.shape}')
    assert predictions.shape == actuals.shape
    return (torch.sum((predictions > 0.5) == actuals) / size) * 100


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 33
    number_of_samples = 1000
    print(device)
    src_pad_idx = 0
    # trg_pad_idx = 0
    src_vocab_size = 26
    number_of_epochs = 50
    learning_rate = 0.001

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
    #     device
    # )
    x = common_utils.load_random_data_samples(number_of_samples)
    y = common_utils.load_random_data_samples(number_of_samples)

    input_sequences = torch.from_numpy(x[0:len(x), :-1]).type(torch.LongTensor)
    results = torch.from_numpy(x[0:len(x), -1]).type(torch.FloatTensor)

    test_sequences = torch.from_numpy(y[0:len(y), :-1]).type(torch.LongTensor)
    test_results = torch.from_numpy(y[0:len(y), -1]).type(torch.FloatTensor)
    # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # print(trg[:, :-1])

    # src_vocab_size,
    # embed_size = 512,
    # num_layers = 6,
    # forward_expansion = 4,
    # heads = 8,
    # dropout = 0,
    # device = "cpu",
    # max_length = 100,
    model = Classifier(seq_len, src_vocab_size, src_pad_idx=src_pad_idx, device=device).to(
        device
    )
    # out = model(input_sequences)
    # print(out.shape)
    # define the optimizer
    # print(model.parameters)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    train_model_single_batch(model, epochs=number_of_epochs, input_sequences=input_sequences, results=results,
                             optim=optim)
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_sequences)
    test_accuracy = get_acc_measure(test_predictions,
                                    test_results.view(-1, test_predictions.size(-1)), number_of_samples)
    print(f'accuracy of the model is: {test_accuracy}%')

# shape of preds: torch.Size([30000, 1]) and shape of actuals: torch.Size([30000, 1])
# accuracy of the model is: 67.3566665649414%
