import numpy as np

class NMF(object):
    def __init__(self, adj, k=32, a=5, b=2, sigma_overline=1, sigma_hat=1, mu_hat=1):
        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.k = k
        self.W = np.random.randn(self.num_nodes, k)
        self.H = np.random.randn(k, self.num_nodes)
        self.M = np.random.randn(self.num_nodes, k)
        self.W = np.abs(self.W)
        self.H = np.abs(self.H)
        self.H = self.W.T
        self.a = a
        self.b = b
        self.sigma_overline = sigma_overline
        self.sigma_hat = sigma_hat
        self.mu_hat = mu_hat
        self.MU = np.abs(np.random.randn(1, k))
        B = np.abs(np.random.randn(k, 1))
        self.B = np.identity(k) * B

    def M_remap_2(self):
        NM = self.M
        NM_MASK_NEG = NM * ((NM < 0) * 1)
        NM_MASK_POS = (NM - 1) * ((NM > 1) * 1)
        NM = NM - NM_MASK_POS
        NM = NM - NM_MASK_NEG
        self.M = NM

    def M_remap_1(self):
        NM = self.M
        NM_MASK = (NM < 0) + (NM > 1)
        NM_MASK_P = NM_MASK * 1
        NM_MASK_N = (NM_MASK - 1) * -1
        NM_1 = NM_MASK_N * NM
        NM_randn = np.random.randn(self.num_nodes, self.k)
        NM_randn_min = NM_randn.min()
        NM_randn_max = NM_randn.max()
        NM_randn = (NM_randn - NM_randn_min) / (NM_randn_max - NM_randn_min)
        NM_randn_1 = NM_randn * NM_MASK_P
        NM_2 = NM_1 + NM_randn_1
        self.M = NM_2

    def forword(self):
        # remap M to [0, 1]
        self.M_remap_1()

        H = self.H
        W = self.W
        M = self.M
        # beta
        B = self.B
        # MU
        MU = self.MU
        # adjacency matrix
        V = self.adj
        # N * N all-one matrix
        L_NN = np.ones([self.num_nodes, self.num_nodes])
        # 1 * N all-one matrix
        L_1N = np.ones([1, self.num_nodes])
        # 1 * K all-one matrix
        L_1K = np.ones([1, self.k])
        # hyperparameter of M
        sigma_overline = self.sigma_overline
        # hyperparameter of MU
        sigma_hat = self.sigma_hat
        # hyperparameter of MU
        mu_hat = self.mu_hat
        # hyperparameter of beta
        a = self.a
        # hyperparameter of beta
        b = self.b
        # number of nodes
        N = self.num_nodes


        W_ETA = W / (1e-5 + L_NN @ H.T - (L_NN @ (H * M.T).T) * M + W @ B)

        W_POS = (V / (1e-5 + W @ H - ((W * M) @ (H * M.T)))) @ H.T - (
                    (V / (1e-5 + W @ H - ((W * M) @ (H * M.T)))) @ (H * M.T).T) * M
        # updated value of W
        W_NEW = W_ETA * W_POS
        W = W_NEW


        H_ETA = H / (1e-5 + W.T @ L_NN - ((W * M).T @ L_NN) * M.T + B @ H)

        H_POS = W.T @ (V / (1e-5 + W @ H - ((W * M) @ (H * M.T)))) - M.T * (
                    (W * M).T @ (V / (1e-5 + W @ H - ((W * M) @ (H * M.T)))))
        # updated value of H
        H_NEW = H_ETA * H_POS
        H = H_NEW

        # updated value of beta
        B_NEW = (N + a - 1) / (0.5 * (np.sum(W * W, axis=0) + np.sum(H * H, axis=1)) + b)

        M_MINUS_MU_POS = ((M - MU) * (((M - MU) > 0) * 1))

        M_MINUS_MU_NEG = ((M - MU) * ((((M - MU) > 0) * 1 - 1) * -1))

        M_ETA = M / (1e-2 + M_MINUS_MU_POS / (1e-2 + sigma_overline ** 2) +
                     ((V / (1e-2 + W @ H - (W * M) @ (H * M.T))) @ (H * M.T).T) * W +
                     ((V / (1e-2 + W @ H - (W * M) @ (H * M.T))) @ (W * M)) * H.T)

        M_POS = (1e-5 + H.T * (L_NN @ (W * M)) + W * (L_NN @ (H.T * M)) - M_MINUS_MU_NEG / (1e-5 + sigma_overline ** 2))
        # updated value of M
        M_NEW = M_ETA * M_POS
        # updated value of MU
        MU_NEW = (L_1N @ M) * ((sigma_hat ** 2) / (1e-5 + N * (sigma_hat ** 2) + (sigma_overline ** 2))) + \
                 (mu_hat * L_1K) * ((sigma_overline ** 2) / (1e-5 + N * (sigma_hat ** 2) + (sigma_overline ** 2)))

        # save results
        self.W = W_NEW
        self.H = H_NEW
        self.MU = MU_NEW
        self.B = np.identity(self.k) * B_NEW
        self.M = M_NEW

        loss = self.nlog_likelihood()
        return loss

    def train(self, n_iter=10):
        for iter in range(n_iter):
            loss = self.forword()
        return loss

    def WHMK_out(self, limit=1e-1):
        # output current value of W, H, M, and K
        t1 = np.sum(self.W, axis=0) > limit
        W = self.W[:, t1]
        H = self.H[t1, :]
        M = self.M[:, t1]
        K = W.shape[1]
        return W, H, M, K

    def WHMK_out_1(self):
        W = self.W
        H = self.H
        M = self.M
        K = W.shape[1]
        return W, H, M, K

    def get_pair_id(self):
        W, _, _, _ = self.WHMK_out()
        pair_id = np.argmax(W, axis=1)
        return pair_id

    def get_core(self):
        W, H, M, K = self.WHMK_out()
        pair_id = self.get_pair_id()
        core_mask = np.zeros([self.num_nodes, K])
        core_mask_neg = np.zeros([self.num_nodes, K])
        for i in range(self.num_nodes):
            core_mask[i][pair_id[i]] = 1
        core_mask_neg[core_mask == 0] = np.inf
        core_mask_neg[core_mask == 1] = 0

        M = core_mask * M
        pair_count = np.sum(core_mask, axis=0)
        M = M[:, pair_count != 0]
        core_mask_neg = core_mask_neg[:, pair_count != 0]
        pair_count = pair_count[pair_count != 0]
        M_ave = np.sum(M, axis=0) / pair_count
        M = M - M_ave * 0.5
        M = M + core_mask_neg

        mask0 = M >= 0
        mask1 = M < 0
        M[mask0] = 0
        M[mask1] = 1
        M = np.sum(M, axis=1)
        return M

    def get_soft_core(self):
        W, H, M, K = self.WHMK_out()
        pair_id = self.get_pair_id()
        core_mask = np.zeros([self.num_nodes, K])
        core_mask_neg = np.zeros([self.num_nodes, K])
        for i in range(self.num_nodes):
            core_mask[i][pair_id[i]] = 1
        core_mask_neg[core_mask == 0] = np.inf
        core_mask_neg[core_mask == 1] = 0

        M = core_mask * M
        pair_count = np.sum(core_mask, axis=0)
        M = M[:, pair_count != 0]
        core_mask_neg = core_mask_neg[:, pair_count != 0]
        pair_count = pair_count[pair_count != 0]
        M_ave = np.sum(M, axis=0) / pair_count
        M = M - M_ave * 0.5
        M = M + core_mask_neg
        M = np.sum(M, axis=1)
        return M

    def nlog_likelihood(self):
        W = self.W
        H = self.H
        M = self.M
        B = self.B
        V = self.adj
        MU = self.MU
        b = self.b
        a = self.a
        sigma_overline = self.sigma_overline
        sigma_hat = self.sigma_hat
        mu_hat = self.mu_hat
        epsilon = 1e-3
        V_hat = W @ H - (W * M) @ (H * M.T)
        with np.errstate(all='ignore'):
            term1 = V * np.log(V/(V_hat + epsilon)) + V_hat
            term1[np.isnan(term1)] = 0
            term2 = 0.5 * ((W * W) @ B - np.log(np.diagonal(B)))
            term3 = 0.5 * ((B @ (H * H)).T - np.log(np.diagonal(B)))
            term4 = b * np.diagonal(B) - (a - 1) * np.log(np.diagonal(B))
            term5 = (M - MU) * (M - MU) / (2 * sigma_overline * sigma_overline)
            term6 = (MU - mu_hat) * (MU - mu_hat) / (2 * sigma_hat * sigma_hat)
        u = np.sum(term1) + np.sum(term2) + np.sum(term3) + np.sum(term4) + np.sum(term5) + np.sum(term6)
        return u