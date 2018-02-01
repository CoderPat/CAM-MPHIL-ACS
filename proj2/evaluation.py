from gcn_d import GCN 
from lgclp import LGCLP
from mlp import MLP 

from data.utils import load_data

for samples in [20, 60, 100, 140, 180, 220]:
    adj, X, Y_train, Y_val, Y_test, train_mask, val_mask, test_mask = load_data("citeseer", samples)
    train_ratio = samples/X.shape[0]
    gcn = GCN()
    lgclp = LGCLP()
    mlp = MLP()

    mlp.fit(X[train_mask, :], Y_train[train_mask, :])
    mlp_acc = mlp.evaluate(X[test_mask, :], Y_test[test_mask, :])
    gcn.fit(X, Y_train, adj, train_mask)
    gcn_acc = (gcn.evaluate(X, Y_test, adj, test_mask))
    lgclp.fit(adj, Y_train)
    lgclp_acc = lgclp.evaluate(Y_test, test_mask)
    print("%% training samples=%f ; mlp_acc=%f, lgclp_acc=%f, gcn_acc=%f" % (train_ratio, mlp_acc, lgclp_acc, gcn_acc))

