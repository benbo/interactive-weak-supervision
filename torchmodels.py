import numpy as np
import torch
from sklearn.utils import resample
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def weight_reset(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()

    
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu2(hidden2)
        output = self.fc3(relu2)
        out = self.sigmoid(output)
        return out


class FeedforwardFlexible(torch.nn.Module):
    def __init__(self, h_sizes, activations):
        super(FeedforwardFlexible, self).__init__()

        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activations[k])

        self.layers.append(torch.nn.Linear(h_sizes[-1], 1))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TorchMLP:
    def __init__(self, h_sizes=[150, 10, 10], activations=[torch.nn.ReLU(), torch.nn.ReLU()], optimizer='Adam',
                 optimparams={}, nepochs=200):
        self.model = FeedforwardFlexible(h_sizes, activations).float()
        self.optimizer = optimizer
        if optimizer == 'Adam':
            if optimparams:
                self.optimparams = optimparams
            else:
                self.optimparams = {'lr': 1e-3, 'weight_decay': 1e-4}

        self.epochs = nepochs
        
    def fit(self, X, Y, batch_size=None, sample_weights=None, device=None):
        tinput = torch.from_numpy(X)
        target = torch.from_numpy(Y.reshape(-1, 1))
        if device is not None:
            tinput = tinput.to(device)
            target = target.to(device)
        tweights = None
        if sample_weights is not None:
            tweights = torch.from_numpy(sample_weights.reshape(-1, 1))
            if device is not None:
                tweights = tweights.to(device)
               
        criterion = torch.nn.BCELoss(reduction='none')
        self.model.apply(weight_reset)
        
        trainX, trainy = tinput, target
        trainweight = None
        if tweights is not None:
            trainweight = tweights

        if self.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.model.parameters(), 
                                          lr=1,
                                          max_iter=400,
                                          max_eval=15000,
                                          tolerance_grad=1e-07,
                                          tolerance_change=1e-04,
                                          history_size=10,
                                          line_search_fn=None)

            def closure():
                optimizer.zero_grad()
                mout = self.model(trainX)
                closs = criterion(mout, trainy)
                if tweights is not None:
                    closs = torch.mul(closs, trainweight).mean()
                else:
                    closs = closs.mean()

                closs.backward()
                return closs
            # only take one step (one epoch)
            optimizer.step(closure)
        else:
            optimizer = None
            if self.optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(), **self.optimparams)
            elif self.optimizer == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(), **self.optimparams)
            lastloss = None
            tolcount = 0
            if batch_size is None:
                for nep in range(self.epochs):

                    out = self.model(trainX)
                    loss = criterion(out, trainy)
                    if tweights is not None:
                        loss = torch.mul(loss, trainweight).mean()
                    else:
                        loss = loss.mean()

                    # early stopping
                    if lastloss is None:
                        lastloss = loss
                    else:
                        if lastloss-loss < 1e-04:
                            tolcount += 1
                        else:
                            tolcount = 0
                        if tolcount > 9:
                            break

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
            else:
                N = trainX.size()[0]
                dostop = False
                for nep in range(self.epochs):
                    permutation = torch.randperm(N)
                    
                    for i in range(0, N, batch_size):
                        optimizer.zero_grad()
                        indices = permutation[i:i+batch_size]
                        batch_x, batch_y = trainX[indices], trainy[indices]
                        
                        out = self.model(batch_x)
                        loss = criterion(out, batch_y)
                        if tweights is not None:
                            batch_weight = trainweight[indices]
                            loss = torch.mul(loss, batch_weight).mean()
                        else:
                            loss = loss.mean()

                        # early stopping
                        if lastloss is None:
                            lastloss = loss
                        else:
                            if lastloss-loss < 1e-04:
                                tolcount += 1
                            else:
                                tolcount = 0
                            if tolcount > 10:
                                dostop = True
                                break

                        loss.backward()

                        optimizer.step()
                    if dostop:
                        break
        
    def predict(self, Xtest, device=None):
        preds = self.predict_proba(Xtest, device)
        return (preds > 0.5).astype(int)
        
    def predict_proba(self, Xtest, device=None):
        with torch.no_grad():
            tXtest = torch.from_numpy(Xtest)
            if device is not None:
                tXtest = tXtest.to(device)
                preds = self.model(tXtest).data.cpu().numpy().flatten()
            else:
                preds = self.model(tXtest).data.numpy().flatten()
        return preds


def applypredict(args):
    model, Xtest = args
    return model(Xtest).data.numpy().flatten()


def applyfit(args):
    # reset model
    model, ix, N, tinput, target, tweights, random_state, whichoptim, epochs, optimparams = args
    criterion = torch.nn.BCELoss(reduction='none')
    model.apply(weight_reset)
    # select indexes
    train_ix = resample(ix, replace=True, n_samples=N, random_state=random_state)
    trainX, trainy = tinput[train_ix], target[train_ix]
    trainweight = None
    optimizer = None
    if tweights is not None:
        trainweight = tweights[train_ix]

    if whichoptim == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), 
                                      lr=0.1,
                                      max_iter=400,
                                      max_eval=15000,
                                      tolerance_grad=1e-07,
                                      tolerance_change=1e-04,
                                      history_size=10,
                                      line_search_fn=None)

        def closure():
            optimizer.zero_grad()
            cout = model(trainX)
            closs = criterion(cout, trainy)
            if tweights is not None:
                closs = torch.mul(closs, trainweight).mean()
            else:
                closs = closs.mean()
            closs.backward()
            return closs

        lloss = 1.0
        cntr = 0
        while lloss > 0.1 and cntr < epochs:
            cntr += 1
            optimizer.step(closure)
            with torch.no_grad():
                out = model(trainX)
                lloss = criterion(out, trainy)
                if tweights is not None:
                    lloss = torch.mul(lloss, trainweight).mean()
                else:
                    lloss = lloss.mean()
                if lloss > 0.1:
                    model.apply(weight_reset)
    else:
        if whichoptim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **optimparams)
        elif whichoptim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **optimparams)
        lastloss = None
        tolcount = 0
        for nep in range(epochs):
            out = model(trainX)
            loss = criterion(out, trainy)
            if tweights is not None:
                loss = torch.mul(loss, trainweight).mean()
            else:
                loss = loss.mean()

            # early stopping
            if lastloss is None:
                lastloss = loss
            else:
                if lastloss-loss < 1e-04:
                    tolcount += 1
                else:
                    tolcount = 0
                if tolcount > 9:
                    break

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


class BaggingWrapperTorch:
    def __init__(self, nfeatures=150, n_estimators=100, njobs=1,
                 optimizer='Adam', optimparams={}, nepochs=200):
        self.n_estimators = n_estimators
        self.device = torch.device("cpu")  # parallel ensemble only implemented on cpu for now
        self.members = [Feedforward(nfeatures, 10).float().to(self.device) for _ in range(n_estimators)]
        self.mpool = None
        self.njobs = njobs
        self.optimizer = optimizer
        
        if optimizer == 'Adam':
            if optimparams:
                self.optimparams = optimparams
            else:
                self.optimparams = {'lr': 1e-3, 'weight_decay': 1e-4}
        
        self.epochs = nepochs
        if njobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            _ = applyfit
            self.mpool = ctx.Pool(self.njobs)
    
    def fit(self, X, Y, sample_weights=None):
        if np.any(np.isnan(Y)):
            raise ValueError('found nan in Y')
        random_states = np.random.randint(0, 2**32 - 1, self.n_estimators)
        tinput = torch.from_numpy(X)
        target = torch.from_numpy(Y.reshape(-1, 1))
        tweights = None
        if sample_weights is not None:
            tweights = torch.from_numpy(sample_weights.reshape(-1, 1))
        N = len(X)
        ix = list(range(N))
        if self.njobs > 1:
            self.mpool.map(applyfit, list((model, ix, N, tinput, target, tweights, rstate,
                           self.optimizer, self.epochs, self.optimparams) for model, rstate in
                           zip(self.members, random_states)))

        else:
            for j, model in enumerate(self.members):
                applyfit((model, ix, N, tinput, target, tweights, random_states[j], self.optimizer,
                          self.epochs, self.optimparams))

    def predict_raw(self, Xtest):
        with torch.no_grad():
            n = Xtest.shape[0]
            tXtest = torch.from_numpy(Xtest)
            predictions = np.zeros((n, self.n_estimators))
            
            if self.njobs > 1:
                for i, pred in enumerate(self.mpool.map(applypredict, list((model, tXtest)
                                                                           for model in self.members))):
                    predictions[:, i] = pred
            else:
                for i, model in enumerate(self.members):
                    predictions[:, i] = model(tXtest).data.numpy().flatten()

            return predictions
    
    def predict(self, Xtest):
        predictions = self.predict_raw(Xtest)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0
        return predictions.mean(1), predictions.std(1)
        
    def predict_proba(self, Xtest):
        predictions = self.predict_raw(Xtest)
        return predictions.mean(1), predictions.std(1)
