import casadi as ca

def NN(u, Network, sigma):
    """Evaluate the PyTorch neural network with input u using CasADi."""
    y = u
    L = len(Network.linears)
    for i in range(L - 1):
        W = ca.DM(Network.linears[i].weight.detach().cpu().numpy())
        b = ca.DM(Network.linears[i].bias.detach().cpu().numpy())
        y = sigma(W @ y + b)  # @ is matrix multiplication

    W = ca.DM(Network.linears[-1].weight.detach().cpu().numpy())
    b = ca.DM(Network.linears[-1].bias.detach().cpu().numpy())
    y = W @ y + b
    return y

def sigmoid_ca(x):
    return 1 / (1 + ca.exp(-x))

def lstm_cell_ca(x_t, h_prev, c_prev, W_ih, W_hh, b, W_fc, b_fc):
    hidden_size = h_prev.shape[0]

    # Compute gates using CasADi matrix operations
    gates = ca.mtimes(W_ih, x_t) + ca.mtimes(W_hh, h_prev) + b

    # Split gates
    i = sigmoid_ca(gates[0:hidden_size])
    f = sigmoid_ca(gates[hidden_size : 2 * hidden_size])
    g = ca.tanh(gates[2 * hidden_size : 3 * hidden_size])
    o = sigmoid_ca(gates[3 * hidden_size : 4 * hidden_size])

    c_t = f * c_prev + i * g
    h_t = o * ca.tanh(c_t)

    # Output layer
    y_t = ca.mtimes(W_fc, h_t) + b_fc

    return y_t, h_t, c_t

def extract_lstm_weights_ca(model):
    """从 PyTorch LSTM 模型中提取权重并转换为 CasADi 矩阵"""
    import casadi as ca
    
    W_ih = ca.DM(model.lstm.weight_ih_l0.detach().cpu().numpy()) 
    W_hh = ca.DM(model.lstm.weight_hh_l0.detach().cpu().numpy()) 
    b_ih = ca.DM(model.lstm.bias_ih_l0.detach().cpu().numpy())   
    b_hh = ca.DM(model.lstm.bias_hh_l0.detach().cpu().numpy())   
    b = b_ih + b_hh

    W_fc = ca.DM(model.fc.weight.detach().cpu().numpy())
    b_fc = ca.DM(model.fc.bias.detach().cpu().numpy())
    
    return W_ih, W_hh, b, W_fc, b_fc


def deepar_cell_ca(x_t, h_prev, c_prev, W_ih, W_hh, b,
                   W_mu1, b_mu1, W_mu2, b_mu2,
                   W_sig1, b_sig1, W_sig2, b_sig2):
    """
    提取DeepAR的均值 mu 和 标准差 sigma
    """
    hidden_size = h_prev.shape[0]
    
    # LSTM Backbone
    gates = ca.mtimes(W_ih, x_t) + ca.mtimes(W_hh, h_prev) + b
    i = sigmoid_ca(gates[0:hidden_size])
    f = sigmoid_ca(gates[hidden_size : 2 * hidden_size])
    g = ca.tanh(gates[2 * hidden_size : 3 * hidden_size])
    o = sigmoid_ca(gates[3 * hidden_size : 4 * hidden_size])

    c_t = f * c_prev + i * g
    h_t = o * ca.tanh(c_t)
    
    # Mu Decoder (Linear -> ReLU -> Linear)
    mu_hidden = ca.fmax(0, ca.mtimes(W_mu1, h_t) + b_mu1)
    mu_t = ca.mtimes(W_mu2, mu_hidden) + b_mu2
    
    # Sigma Decoder (Linear -> ReLU -> Linear -> Softplus)
    sig_hidden = ca.fmax(0, ca.mtimes(W_sig1, h_t) + b_sig1) # ReLU
    sig_raw = ca.mtimes(W_sig2, sig_hidden) + b_sig2
    sigma_t = ca.log(1 + ca.exp(sig_raw)) + 1e-3
    
    return mu_t, sigma_t, h_t, c_t

def extract_deepar_weights_ca(model):
    """
    提取DeepAR模型权重
    """
    #LSTM
    W_ih = ca.DM(model.lstm.weight_ih_l0.detach().cpu().numpy()) 
    W_hh = ca.DM(model.lstm.weight_hh_l0.detach().cpu().numpy()) 
    b = ca.DM((model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0).detach().cpu().numpy())
    # Mu Decoder
    W_mu1 = ca.DM(model.mu_decoder[0].weight.detach().cpu().numpy())
    b_mu1 = ca.DM(model.mu_decoder[0].bias.detach().cpu().numpy())
    W_mu2 = ca.DM(model.mu_decoder[2].weight.detach().cpu().numpy())
    b_mu2 = ca.DM(model.mu_decoder[2].bias.detach().cpu().numpy())
    # Sigma Decoder
    W_sig1 = ca.DM(model.sigma_decoder[0].weight.detach().cpu().numpy())
    b_sig1 = ca.DM(model.sigma_decoder[0].bias.detach().cpu().numpy())
    W_sig2 = ca.DM(model.sigma_decoder[2].weight.detach().cpu().numpy())
    b_sig2 = ca.DM(model.sigma_decoder[2].bias.detach().cpu().numpy())
    
    return W_ih, W_hh, b, W_mu1, b_mu1, W_mu2, b_mu2, W_sig1, b_sig1, W_sig2, b_sig2