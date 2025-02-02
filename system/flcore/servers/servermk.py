import copy
from collections import OrderedDict

import numpy as np
import torch
import time
from system.flcore.clients.clientmk import *
from system.utils.data_utils import read_client_data
from system.flcore.trainmodel.models import CNNHyper
from threading import Thread


class pFedMK:
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_modules = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        in_dim = list(args.model.head.parameters())[0].shape[1]
        cs = ConditionalSelection(in_dim, in_dim).to(args.device)
        self.device = args.device
        self.num_classes = args.num_classes
        self.hnet = CNNHyper(n_nodes=self.num_clients, embedding_dim=128, dim=512, num_classes=200, hidden_dim=100,
                             n_hidden=3, client_sample=self.num_join_clients).to(self.device)
        self.optimizer = torch.optim.SGD(
            params=self.hnet.parameters(), lr=0.005)
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientMK(args,
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            ConditionalSelection=cs)
            self.clients.append(client)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.head = None
        self.cs = None


    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_modules)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_modules.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_modules = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_modules.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def evaluate(self, acc=None):
        stats = self.test_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.hnet.train()
            h_grad = []
            self.selected_clients = self.select_clients()
            self.selected_clients_indices = [self.clients.index(client) for client in self.selected_clients]
            self.weights = self.hnet(torch.tensor([self.selected_clients_indices], dtype=torch.long).to(self.device),
                                     False)
            grads_update = [torch.zeros_like(p) for p in self.hnet.parameters()]

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate before local training")
                self.evaluate()

            for client in self.selected_clients:
                client.train_cs_model()
                client.generate_upload_head()

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()
            self.global_head()
            self.global_cs()
            for client, ix in zip(self.selected_clients, range(len(self.selected_clients_indices))):
                final_state = client.model.head_g.state_dict()
                node_weights = self.weights[ix]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), self.hnet.parameters(), grad_outputs=list(delta_theta.values()),
                    retain_graph=True)
                h_grad.append(hnet_grads)
            for w, hgrad in zip(self.uploaded_weights, h_grad):
                for i, grad in enumerate(hgrad):
                    grads_update[i] += w * grad
            self.optimizer.zero_grad()
            for p, g in zip(self.hnet.parameters(), grads_update):
                p.grad = g
            self.optimizer.step()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.model.feature_extractor)

    def global_head(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.head_g)

        self.head = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.head.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_head(w, client_model)

        for client, ix in zip(self.selected_clients, range(len(self.selected_clients_indices))):
            client.set_head_g(self.head)
            client.set_head_hp(self.head, self.weights[ix])

    def add_head(self, w, head):
        for server_param, client_param in zip(self.head.parameters(), head.parameters()):
            server_param.data += client_param.data.clone() * w
            
    def global_cs(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.gate.cs)

        self.cs = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.cs.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_cs(w, client_model)

        for client in self.selected_clients:
            client.set_cs(self.cs)

    def add_cs(self, w, cs):
        for server_param, client_param in zip(self.cs.parameters(), cs.parameters()):
            server_param.data += client_param.data.clone() * w


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()
        
        self.fc = nn.Sequential(
            nn.LayerNorm([h_dim]),
            nn.Linear(in_dim, h_dim*2),
            #nn.LayerNorm([h_dim*2]), # Use this normalization when using smaller-scale datasets like Cifar, as it is less likely to diverge.
            nn.ReLU(),
        )
        self.conv = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1,
                              bias=True, padding=0)

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = self.conv(x)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]
