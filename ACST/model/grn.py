import torch
import torch.nn as nn


class GLU(nn.Module):
    '''
    门控线性单元
    '''

    def __init__(self, input_size):
        super(GLU, self).__init__()
        # 输入
        self.a = nn.Linear(input_size, input_size, bias=True)
        self.b = nn.Linear(input_size, input_size, bias=True)
        self.sigmoid = nn.Sigmoid()

        pass

    def forward(self, x):
        # 门控
        return torch.mul(self.sigmoid(self.b(x)), self.a(x))


class TemporalLayer(nn.Module):
    '''
    时域层
    '''

    def __init__(self, module):
        super(TemporalLayer, self).__init__()
        self.module = module

        pass

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.reshape(t, n, x.size(-1))
        return x


class GatedResidualNetwork(nn.Module):
    '''
    门控残差单元
    (跳过架构中任何未使用的组件，提供自适应深度和网络复杂性)
    '''

    def __init__(self, input_size, hidden_size, output_size, dropout=0.2, context_size=None,
                 is_temporal=False):
        '''
        初始化函数
        :param input_size: 输入维度
        :param hidden_size: 隐藏维度
        :param output_size: 输出维度
        :param dropout: 丢失率
        :param num_lstm_layers: LSTM编码层的层数
        :param context_size: 额外上下文维度
        :param is_temporal: 是否是时域信息
        '''
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.context_size = context_size
        self.is_temporal = is_temporal

        if self.is_temporal:
            if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size, self.output_size))

            # 上下文向量c
            if self.context_size != None:
                self.c = TemporalLayer(nn.Linear(self.context_size, self.hidden_size, bias=False))

            # 第一个线性层和ELU激活函数
            self.dense1 = TemporalLayer(nn.Linear(self.input_size, self.hidden_size))
            self.elu = nn.ELU()

            # 第二个线性层和丢失层
            self.dense2 = TemporalLayer(nn.Linear(self.hidden_size, self.output_size))
            self.dropout = nn.Dropout(p=self.dropout)

            # 门控单元和归一化层
            self.gate = TemporalLayer(GLU(self.output_size))
            self.layer_norm = TemporalLayer(nn.BatchNorm1d(self.output_size))
            pass
        else:
            if self.input_size != self.output_size:
                self.skip_layer = nn.Linear(self.input_size, self.output_size)

            # 上下文向量c
            if self.context_size != None:
                self.c = nn.Linear(self.context_size, self.hidden_size, bias=False)

            # 第一个线性层和ELU激活函数
            self.dense1 = nn.Linear(self.input_size, self.hidden_size)
            self.elu = nn.ELU()

            # 第二个线性层和丢失层
            self.dense2 = nn.Linear(self.hidden_size, self.output_size)
            self.dropout = nn.Dropout(p=self.dropout)

            # 门控单元和归一化层
            self.gate = GLU(self.output_size)
            self.layer_norm = nn.BatchNorm1d(self.output_size)
            pass

        pass

    def forward(self, x, c=None):
        if self.input_size != self.output_size:
            a = self.skip_layer(x)
        else:
            a = x
        x = self.dense1(x)
        if c != None:
            c = self.c(c.unsqueeze(1))
            x += c
        eta_2 = self.elu(x)
        eta_1 = self.dense2(eta_2)
        eta_1 = self.dropout(eta_1)

        gate = self.gate(eta_1)
        gate += a
        # x = self.layer_norm(gate)
        x = gate

        return x
