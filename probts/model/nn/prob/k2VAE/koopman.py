'''
* @author: EmpyreanMoon
*
* @create: 2025-01-02 14:10
*
* @description: The koopman encoder & decoder
'''
import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """

    def __init__(self, dynamic_dim, mode):
        super(KPLayer, self).__init__()
        self.mode = mode
        if self.mode != 'static':
            K_init = torch.randn(dynamic_dim, dynamic_dim)
            U, _, V = torch.svd(K_init)  # stable initialization
            self.K_dynamic = nn.Parameter(torch.mm(U, V.t()))
        self.K_static = None  # B E E

    def one_step_forward(self, z, return_rec=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        if self.mode == 'static':
            self.K = torch.linalg.lstsq(x, y).solution  # B E E
        elif self.mode == 'dynamic':
            self.K = self.K_dynamic.unsqueeze(0).repeat(B, 1, 1)
        else:
            self.K = torch.linalg.lstsq(x, y).solution + self.K_dynamic.unsqueeze(0).repeat(B, 1, 1)

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred

    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """

    def __init__(self, dynamic_dim, mode):
        super(KPLayerApprox, self).__init__()
        self.mode = mode
        if self.mode != 'static':
            K_init = torch.randn(dynamic_dim, dynamic_dim)
            U, _, V = torch.svd(K_init)  # stable initialization
            self.K_dynamic = nn.Parameter(torch.mm(U, V.t()))
        self.K = None  # B E E
        self.K_step = None  # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        if self.mode == 'static':
            self.K = torch.linalg.lstsq(x, y).solution  # B E E
        elif self.mode == 'dynamic':
            self.K = self.K_dynamic.unsqueeze(0).repeat(B, 1, 1)
        else:
            self.K = torch.linalg.lstsq(x, y).solution + self.K_dynamic.unsqueeze(0).repeat(B, 1, 1)

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # B L E

        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred
