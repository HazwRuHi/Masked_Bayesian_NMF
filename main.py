import argparse
from model import NMF
from utils import randnet, nmi_cp

parser = argparse.ArgumentParser(description='LinkPrediction')
parser.add_argument('--network_size', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--a', type=int, default=5)
parser.add_argument('--b', type=int, default=10)
parser.add_argument('--sigma_overline', type=int, default=1)
parser.add_argument('--sigma_hat', type=int, default=1)
parser.add_argument('--mu_hat', type=int, default=1)
parser.add_argument('--k', type=int, default=32)

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    adj, p_label, c_label, n_pair = randnet(args.network_size)
    model = NMF(adj, k=args.k, a=args.a, b=args.b,
                sigma_overline=args.sigma_overline,
                sigma_hat=args.sigma_hat,
                mu_hat=args.mu_hat)
    model.train(n_iter=args.epochs)

    c_pred = model.get_core()
    p_pred = model.get_pair_id()
    nmicp = nmi_cp(c_label, c_pred, p_label, p_pred)

    print("NMI_CP:{}".format(nmicp))
    print("done")

