import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0,bidirectional=False,wdrop=0.5):
        super(LSTM, self).__init__()
        if bidirectional:
            raise NotImplementedError()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.wdrop = wdrop
        self.lstm_cells = [LSTMCell(self.input_size,self.hidden_size,bias=self.bias,wdrop=self.wdrop) if i == 0 else LSTMCell(self.hidden_size,self.hidden_size,bias=self.bias,wdrop=self.wdrop) for i in range(self.num_layers)]
        self.lstm_cells = torch.nn.ModuleList(self.lstm_cells)

    def init_hidden(self,batch_size):
        if self.batch_first:
            return (torch.zeros((batch_size,self.num_layers,self.hidden_size)),torch.zeros((batch_size,self.num_layers,self.hidden_size)))
        else:
            return (torch.zeros((self.num_layers,batch_size,self.hidden_size)),torch.zeros((self.num_layers,batch_size,self.hidden_size)))
    def forward(self,x,hidden_states):
        if self.batch_first:
            seqlen = x.size(1)
        else:
            seqlen = x.size(0)
        def layer_operation(x,num_layer,lstm_cell):
            hs = []
            cs = []
            index = [i for i in range(seqlen)]
            hidden = (hidden_states[0][num_layer],hidden_states[1][num_layer])
            for i in index:
                if self.batch_first:
                    inputx = x[:,i,:]
                else:
                    inputx = x[i,:,:]
                _, hidden = lstm_cell(inputx,hidden)
                hs.append(hidden[0])
                cs.append(hidden[1])
            if self.batch_first:
                hs = torch.stack(hs,dim=1)
                cs = torch.stack(cs,dim=1)
            else:
                hs = torch.stack(hs,dim=0)
                cs = torch.stack(cs,dim=0)
            return hs, cs
        
        output_hs = []
        output_cs = []

        for num_layer, lstm_cell in enumerate(self.lstm_cells):
            #forward direction
            hs, cs = layer_operation(x,num_layer,lstm_cell)
            if num_layer != self.num_layers-1 and self.dropout:
                hs = self.dropout_layer(hs)
                cs = self.dropout_layer(cs)
            x = hs
            if batch_first:
                output_hs.append(hs[:,-1,:])
                output_cs.append(cs[:,-1,:])
            else:
                output_hs.append(hs[-1])
                output_cs.append(hs[-1])
        output_hs = torch.stack(output_hs,dim=0)
        output_cs = torch.stack(output_cs,dim=0)
        return hs, (output_hs,output_cs)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, wdrop=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.wdrop = wdrop
        self.i2h = nn.Linear(input_size, 4*hidden_size,bias=self.bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size,bias=self.bias)
        if self.wdrop > 0.:
            self.h2h.weight = nn.Parameter(F.dropout(self.h2h.weight,p=self.wdrop, training=self.training))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        nn.ModuleList([self.i2h,self.h2h])
        self.reset_parameters()

    def init_hidden(self,batch_size):
        return (torch.zeros(batch_size,self.hidden_size),torch.zeros(batch_size,self.hidden_size))
    
    def sample_mask(self):
        keep = 1.0 -self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)

    def forward(self, x, hidden):
        h, c = hidden
        preact = self.i2h(x) + self.h2h(h)
        inputgate, forgetgate, cellgate,  outputgate = torch.chunk(preact,4,dim=-1)
        inputgate = self.sigmoid(inputgate)
        cellgate = self.tanh(cellgate)
        forgetgate = self.sigmoid(forgetgate)
        outputgate = self.sigmoid(outputgate)
        c = c*forgetgate + inputgate * cellgate
        h = outputgate * self.tanh(c)
        
        return h, (h,c)

if __name__ == '__main__':
    seqlen = 50
    features = 1000
    batchsize = 80
    hiddendim = 300
    num_layers = 3
    batch_first = True
    
    if batch_first:
        x = torch.zeros(batchsize,seqlen,features)
    else:
        x = torch.zeros(seqlen,batchsize,features)
    lstm = LSTM(features,hiddendim,num_layers=num_layers,batch_first=batch_first,wdrop=0.)
    lstm.cuda()
    true_lstm = nn.LSTM(features,hiddendim,num_layers=num_layers,batch_first= batch_first)
    hidden = (torch.zeros(num_layers,batchsize,hiddendim),torch.zeros(num_layers,batchsize,hiddendim))
    for i in range(len(lstm.lstm_cells)):
        lstm.lstm_cells[i].i2h.weight = getattr(true_lstm,'weight_ih_l%i'%i)
        lstm.lstm_cells[i].h2h.weight = getattr(true_lstm,'weight_hh_l%i'%i)
        lstm.lstm_cells[i].i2h.bias = getattr(true_lstm,'bias_ih_l%i'%i)
        lstm.lstm_cells[i].h2h.bias = getattr(true_lstm,'bias_hh_l%i'%i)
    print(lstm.lstm_cells)
    a,b = lstm(x,hidden)
    true_a, true_b = true_lstm(x,hidden)
    print('LSTM output size %s'%str(a.size()))
    print('Expected LSTM output size %s'%str(true_a.size()))
    print('LSTM h_n size %s'%str(b[0].size()))
    print('Expected LSTM h_n size %s'%str(true_b[0].size()))
    print('LSTM c_n size %s'%str(b[1].size()))
    print('Expected LSTM c_n size %s'%str(true_b[1].size()))
    print('Squared difference of output %f'%torch.sum(((a-true_a)**2)))
    print('Squared difference of h_n %f'%torch.sum(((b[0]-true_b[0])**2)))
    print('Squared difference of c_n %f'%torch.sum(((b[1]-true_b[1])**2)))
