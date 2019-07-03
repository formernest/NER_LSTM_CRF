import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # self.hidden = self.init_hidden()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.hidden_dim // 2),
                torch.randn(2, size, self.hidden_dim // 2))

    def _forward_alg(self, all_feats):    # batch*seq_len*tag_size
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((len(all_feats), self.tagset_size), -10000.)
        # START_TAG has all of the score.
        for index in range(len(all_feats)):
            init_alphas[index][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas
        forward_var = init_alphas.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Iterate through the sentence
        alphas = []
        for feats in all_feats:
            for feat in feats:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size).cuda()
                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1).cuda()
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp
                    print("type of trans_score:", type(trans_score))
                    print("type of emit_score:", type(emit_score))
                    print("type of forward_var:", type(forward_var))
                    next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            alpha = log_sum_exp(terminal_var)
            alphas.append(alpha)
        return alphas

    def _get_lstm_features(self, sentence):
        # self.hidden = self.init_hidden(len(sentence))
        embeds = self.word_embeds(sentence)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # batch, seq_len, hidden_size * num_directions
        lstm_out, self.hidden = self.lstm(embeds)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):    # batch*seq_len*tag_size, batch*seq_len
        # Gives the score of a provided tag sequence
        score = torch.zeros(len(feats), 1)
        padding = [self.tag_to_ix[START_TAG]] * len(feats)
        padding = torch.tensor(padding, dtype=torch.long)
        padding = padding.view(-1, 1)
        # padding = torch.Tensor([self.tag_to_ix[START_TAG]] * self.batch_size, dtype=torch.long)
        tags = torch.cat((padding, tags), 1)
        for batch, feat in enumerate(feats):
            # print("feats:", feats)
            # print("batch:", batch, type(batch))
            # print("feat:", feat, len(feat))  # seq_len*tag_size
            for i in range(len(feat)):
                score[batch] += self.transitions[tags[batch][i + 1], tags[batch][i]] + feat[i][tags[batch][i + 1]]
            score[batch] = score[batch] + self.transitions[self.tag_to_ix[STOP_TAG], tags[batch][-1]]
        return score

    def _viterbi_decode(self, feats):  # torch.Size([64, 60, 9])  batch*seq_len*tag_size
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)   # 1 * self.tagset_size
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        path_score_all = list()
        best_path_all = list()
        for batch, feats in enumerate(feats):
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            path_score_all.append(path_score)
            best_path_all.append(best_path)
        return path_score_all, best_path_all

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)    # batch*seq_len*tag_size
        forward_score = self._forward_alg(feats)     # crf的计算
        gold_score = self._score_sentence(feats, tags)   #
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)    # batch*seq_len*tag_size  每个字符映射到每个tag的发射概率
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
