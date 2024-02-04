import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph

def normalize_Hyper(H):
    D_v = sp.diags(1 / (np.sqrt(H.sum(axis=1).A.ravel()) + 1e-8))
    D_e = sp.diags(1 / (np.sqrt(H.sum(axis=0).A.ravel()) + 1e-8))
    H_nomalized = D_v @ H @ D_e @ H.T @ D_v
    return H_nomalized

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph

def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class DHBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.hyper_num = conf["hyper_num"] 

        self.num_layers = self.conf["num_layers"]

        self.mlp = conf["mlp"]
        self.h_p = conf["h_p"]
        self.loss_n_lambda = conf["loss_n_lambda"]
        self.loss_ttl_lambda = conf["loss_ttl_lambda"]
        self.b_u_loss = conf["b_u_loss"]

        self.init_emb()
        self.init_hyper() 

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()

        self.c_temp = self.conf["c_temp"]
        
    def init_hyper(self):
        self.IL_user_hyper = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.embedding_size, self.hyper_num)))
        self.IL_item_hyper = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.embedding_size, self.hyper_num)))
        self.BL_bundle_hyper = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.embedding_size, self.hyper_num)))
        self.BL_user_hyper = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.embedding_size, self.hyper_num)))
        
        self.act = nn.LeakyReLU(negative_slope=0.05)
        
        self.IL_mlp_users = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.IL_mlp_items = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.BL_mlp_bundles = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.BL_mlp_users = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])

        self.hyper_weight1 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.hyper_weight2 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.hyper_weight3 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))

        self.BL_user_hyper_weight1 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.BL_user_hyper_weight2 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.BL_user_hyper_weight3 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.BL_bundle_hyper_weight1 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.BL_bundle_hyper_weight2 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.BL_bundle_hyper_weight3 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))

    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

        self.IL_user_bound = nn.Parameter(torch.FloatTensor(self.embedding_size, 1))
        nn.init.xavier_normal_(self.IL_user_bound)
        self.BL_user_bound = nn.Parameter(torch.FloatTensor(self.embedding_size, 1))
        nn.init.xavier_normal_(self.BL_user_bound)
        

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED": #edge dropout
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)

    def hyper_Propagate(self, feature, adj, weight1_hyper, weight2_hyper, weight3_hyper):
        feature1 = self.act(adj.T @ feature)
        feature2 = self.act(weight1_hyper @ feature1) + feature1
        if self.h_p == 1 :
            feature = self.act(adj @ feature2)
            return feature
        feature3 = self.act(weight2_hyper @ feature2) + feature2
        if self.h_p == 2 :
            feature = self.act(adj @ feature3)
        else:    
            feature4 = self.act(weight3_hyper @ feature3) + feature3
            feature = self.act(adj @ feature4)
        return feature

    def IL_propagate(self, graph, users_feature, items_feature, mess_dropout, test): 
        features = torch.cat((users_feature, items_feature), 0)
        all_features = [features]
        all_hyper_features = [features]
        all_fixed_features = [features + features] 
        for i in range(self.num_layers):
            features = torch.spmm(graph, features) #GCN
            if self.conf["aug_type"] == "MD" and not test: 
                features = mess_dropout(features)

            features = F.normalize(features, p=2, dim=1)
            users_feature, items_feature = torch.split(features, (users_feature.shape[0], items_feature.shape[0]), 0)


            
            if(self.mlp == 2):
                users_hypergraph = self.IL_mlp_users[i](users_feature)
                items_hypergraph = self.IL_mlp_items[i](items_feature)
            else:
                users_hypergraph = nn.Parameter(torch.FloatTensor(self.num_users, self.hyper_num)).to(self.device)
                nn.init.xavier_normal_(users_hypergraph)
                items_hypergraph = nn.Parameter(torch.FloatTensor(self.num_items, self.hyper_num)).to(self.device)
                nn.init.xavier_normal_(items_hypergraph)

            if(self.conf["aug_type"] == "ED"):
                hyper_ratio = self.conf["item_level_ratio"]
                users_hypergraph = F.dropout(users_hypergraph, p = 1 - hyper_ratio)
                items_hypergraph = F.dropout(items_hypergraph, p = 1 - hyper_ratio)

            if(self.h_p != 0):
                hyper_users_feature = self.hyper_Propagate(users_feature, users_hypergraph, self.hyper_weight1, self.hyper_weight2, self.hyper_weight3)
                hyper_items_feature = self.hyper_Propagate(items_feature, items_hypergraph, self.hyper_weight1, self.hyper_weight2, self.hyper_weight3)
            else:
                hyper_users_feature = self.act(users_hypergraph @ self.act(users_hypergraph.permute(1, 0) @ users_feature))
                hyper_items_feature = self.act(items_hypergraph @ self.act(items_hypergraph.permute(1, 0) @ items_feature))

            hyper_features = torch.cat((hyper_users_feature, hyper_items_feature), 0)

            if self.conf["aug_type"] == "MD" and not test:
                hyper_features = mess_dropout(features)

            hyper_features = F.normalize(hyper_features, p=2, dim=1)

            fixed_features = (features + hyper_features)
            fixed_features = F.normalize(fixed_features, p=2, dim=1)
            all_features.append(features)
            all_hyper_features.append(hyper_features)
            all_fixed_features.append(fixed_features)


        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)
        all_features = all_features / (self.num_layers + 1)

        all_hyper_features = torch.stack(all_hyper_features, 1)
        all_hyper_features = torch.sum(all_hyper_features, dim=1).squeeze(1)
        all_hyper_features = all_hyper_features / (self.num_layers + 1)

        all_fixed_features = torch.stack(all_fixed_features, 1)
        all_fixed_features = torch.sum(all_fixed_features, dim=1).squeeze(1)
        all_fixed_features = all_fixed_features / (self.num_layers + 1)

        ori_users_feature, ori_items_feature = torch.split(all_features, (users_feature.shape[0], items_feature.shape[0]), 0) 
        hyper_users_feature, hyper_items_feature = torch.split(all_hyper_features, (users_feature.shape[0], items_feature.shape[0]), 0)
        fixed_users_feature, fixed_items_feature = torch.split(all_fixed_features, (users_feature.shape[0], items_feature.shape[0]), 0)

        return ori_users_feature, ori_items_feature, hyper_users_feature, hyper_items_feature, fixed_users_feature, fixed_items_feature

    def BL_propagate(self, graph, users_feature, bundles_feature, mess_dropout, test): 
        features = torch.cat((users_feature, bundles_feature), 0)
        all_features = [features]
        all_hyper_features = [features]
        all_fixed_features = [features + features] 
        
        for i in range(self.num_layers):
            features = torch.spmm(graph, features) #GCN
            if self.conf["aug_type"] == "MD" and not test:
                features = mess_dropout(features)

            features = F.normalize(features, p=2, dim=1)
            users_feature, bundles_feature = torch.split(features, (users_feature.shape[0], bundles_feature.shape[0]), 0)
            
            
            if(self.mlp == 2):
                users_hypergraph = self.BL_mlp_users[i](users_feature)
                bundles_hypergraph = self.BL_mlp_bundles[i](bundles_feature)
            else:
                users_hypergraph = nn.Parameter(torch.FloatTensor(self.num_users, self.hyper_num)).to(self.device)
                nn.init.xavier_normal_(users_hypergraph)
                bundles_hypergraph = nn.Parameter(torch.FloatTensor(self.num_bundles, self.hyper_num)).to(self.device)
                nn.init.xavier_normal_(bundles_hypergraph)
        
            
            if(self.conf["aug_type"] == "ED"):
                hyper_ratio = self.conf["bundle_level_ratio"]
                users_hypergraph = F.dropout(users_hypergraph, p = 1 - hyper_ratio)
                bundles_hypergraph = F.dropout(bundles_hypergraph, p = 1 - hyper_ratio)
            if(self.h_p != 0):
                hyper_users_feature = self.hyper_Propagate(users_feature, users_hypergraph, self.BL_user_hyper_weight1, self.BL_user_hyper_weight2, self.BL_user_hyper_weight3)
                hyper_bundles_feature = self.hyper_Propagate(bundles_feature, bundles_hypergraph, self.BL_bundle_hyper_weight1, self.BL_bundle_hyper_weight2, self.BL_bundle_hyper_weight3)
            else:
                hyper_users_feature = self.act(users_hypergraph @ self.act(users_hypergraph.permute(1, 0) @ users_feature))
                hyper_bundles_feature = self.act(bundles_hypergraph @ self.act(bundles_hypergraph.permute(1, 0) @ bundles_feature))

            hyper_features = torch.cat((hyper_users_feature, hyper_bundles_feature), 0)

            if self.conf["aug_type"] == "MD" and not test:
                hyper_features = mess_dropout(features)

            hyper_features = F.normalize(hyper_features, p=2, dim=1)

            fixed_features = (features + hyper_features)#+all_fixed_features[-1]
            fixed_features = F.normalize(fixed_features, p=2, dim=1)
            all_features.append(features)
            all_hyper_features.append(hyper_features)
            all_fixed_features.append(fixed_features)


        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)
        all_features = all_features / (self.num_layers + 1)

        all_hyper_features = torch.stack(all_hyper_features, 1)
        all_hyper_features = torch.sum(all_hyper_features, dim=1).squeeze(1)
        all_hyper_features = all_hyper_features / (self.num_layers + 1)

        all_fixed_features = torch.stack(all_fixed_features, 1)
        all_fixed_features = torch.sum(all_fixed_features, dim=1).squeeze(1)
        all_fixed_features = all_fixed_features / (self.num_layers + 1)

        ori_users_feature, ori_bundles_feature = torch.split(all_features, (users_feature.shape[0], bundles_feature.shape[0]), 0) 
        hyper_users_feature, hyper_bundles_feature = torch.split(all_hyper_features, (users_feature.shape[0], bundles_feature.shape[0]), 0)
        fixed_users_feature, fixed_bundles_feature = torch.split(all_fixed_features, (users_feature.shape[0], bundles_feature.shape[0]), 0)

        return ori_users_feature, ori_bundles_feature, hyper_users_feature, hyper_bundles_feature, fixed_users_feature, fixed_bundles_feature

    def get_IL_bundle_rep(self, IL_items_feature, test):
        
        #att_score = torch.matmul(IL_items_feature, self.IL_bundle_weight)
        #IL_items_feature = torch.matmul(att_score, IL_items_feature)

        if test:
            #graph = F.normalize(self.bundle_agg_graph_ori, p=1, dim=2)
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            #graph = F.normalize(self.bundle_agg_graph, p=1, dim=2)
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):
        if test:
            IL_ori_users_feature, IL_ori_items_feature, IL_hyper_users_feature, IL_hyper_items_feature, IL_fixed_users_feature, IL_fixed_items_feature \
                = self.IL_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            IL_ori_users_feature, IL_ori_items_feature, IL_hyper_users_feature, IL_hyper_items_feature, IL_fixed_users_feature, IL_fixed_items_feature \
                = self.IL_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)

        IL_ori_bundles_feature = self.get_IL_bundle_rep(IL_ori_items_feature, test)
        IL_hyper_bundles_feature = self.get_IL_bundle_rep(IL_hyper_items_feature, test)
        IL_fixed_bundles_feature = self.get_IL_bundle_rep(IL_fixed_items_feature, test)

        if test:
            BL_ori_users_feature, BL_ori_bundles_feature, BL_hyper_users_feature, BL_hyper_bundles_feature, BL_fixed_users_feature, BL_fixed_bundles_feature \
                = self.BL_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_ori_users_feature, BL_ori_bundles_feature, BL_hyper_users_feature, BL_hyper_bundles_feature, BL_fixed_users_feature, BL_fixed_bundles_feature \
                = self.BL_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)

        ori_users_feature = [IL_ori_users_feature, BL_ori_users_feature]
        ori_bundles_feature = [IL_ori_bundles_feature, BL_ori_bundles_feature]

        hyper_users_feature = [IL_hyper_users_feature, BL_hyper_users_feature]
        hyper_bundles_feature = [IL_hyper_bundles_feature, BL_hyper_bundles_feature]

        fixed_users_feature = [IL_fixed_users_feature, BL_fixed_users_feature]
        fixed_bundles_feature = [IL_fixed_bundles_feature, BL_fixed_bundles_feature]


        return ori_users_feature, ori_bundles_feature,hyper_users_feature, hyper_bundles_feature, fixed_users_feature, fixed_bundles_feature


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :] 
        aug = aug[:, 0, :] 

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(self.loss_ttl_lambda *  ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))
        return c_loss

    def cal_diff_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :] 
        aug = aug[:, 0, :] 

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        #pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        #pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(1 / ttl_score))

        return c_loss
    
    def cal_allign_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :] 
        aug = aug[:, 0, :] 

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        #ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        #ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score))

        return c_loss
    
    def cal_UIB_loss(self, pred, user_bound):
        # pred: [bs, 1+neg_num]
        
        loss_p = -torch.log(torch.sigmoid(pred[:, :1] - user_bound))
        loss_n = -torch.log(torch.sigmoid(user_bound - pred[:, 1:]))
        loss = loss_p + self.loss_n_lambda * loss_n
        loss = torch.mean(loss)
        return loss


        
    def cal_loss(self, users_feature, bundles_feature):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature
        # [bs, 1+neg_num]
        
        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)
        
        if(self.b_u_loss == 0):
            #bpr_loss
            loss = cal_bpr_loss(pred)
        else:
            #UIBloss
            IL_user_score_bound = IL_users_feature[:, 0, :].squeeze(1) @ self.IL_user_bound
            BL_user_score_bound = BL_users_feature[:, 0, :].squeeze(1) @ self.BL_user_bound
            user_score_bound = IL_user_score_bound + BL_user_score_bound
            loss = self.cal_UIB_loss(pred, user_score_bound)

        # cl is abbr. of "contrastive loss"
        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return loss, c_loss


    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        ori_users_feature, ori_bundles_feature,hyper_users_feature, hyper_bundles_feature, fixed_users_feature, fixed_bundles_feature = self.propagate()
        
        ori_users_feature = [i[users].expand(-1, bundles.shape[1], -1) for i in ori_users_feature]
        IL_ori_users_feature, BL_ori_users_feature = ori_users_feature

        ori_bundles_feature = [i[bundles] for i in ori_bundles_feature]
        IL_ori_bundles_feature, BL_ori_bundles_feature = ori_bundles_feature

        hyper_users_feature = [i[users].expand(-1, bundles.shape[1], -1) for i in hyper_users_feature]
        IL_hyper_users_feature, BL_hyper_users_feature = hyper_users_feature

        hyper_bundles_feature = [i[bundles] for i in hyper_bundles_feature]
        IL_hyper_bundles_feature, BL_hyper_bundles_feature = hyper_bundles_feature

        IL_users_c_loss = self.cal_c_loss(IL_ori_users_feature, IL_hyper_users_feature)
        IL_bundles_c_loss = self.cal_c_loss(IL_ori_bundles_feature, IL_hyper_bundles_feature)
        BL_users_c_loss = self.cal_c_loss(BL_ori_users_feature, BL_hyper_users_feature)
        BL_bundles_c_loss = self.cal_c_loss(BL_ori_bundles_feature, BL_hyper_bundles_feature)
        
        oh_c_losses = [IL_users_c_loss + BL_users_c_loss, IL_bundles_c_loss + BL_bundles_c_loss]
        oh_c_loss = sum(oh_c_losses) / len(oh_c_losses)

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in fixed_users_feature] 
        bundles_embedding = [i[bundles] for i in fixed_bundles_feature]
        
        #uib_loss, c_loss= self.cal_loss(ori_users_feature, ori_bundles_feature)
        uib_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)
        
        return uib_loss, c_loss, oh_c_loss


    def evaluate(self, propagate_result, users):
        _, _, _, _, users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores
