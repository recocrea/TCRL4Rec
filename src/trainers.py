# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        HIT_50, NDCG_50, MRR = get_metric(pred_list, 50)
        post_fix = {
            "Epoch": epoch,
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "HIT@50": '{:.4f}'.format(HIT_50), "NDCG@50": '{:.4f}'.format(NDCG_50),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_10, NDCG_10, HIT_50, NDCG_50, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [10, 50]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@10": '{:.4f}'.format(recall[0]), "NDCG@10": '{:.4f}'.format(ndcg[0]),
            "HIT@50": '{:.4f}'.format(recall[1]), "NDCG@50": '{:.4f}'.format(ndcg[1])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def get_content_branch_full_sort_score(self, epoch, answers, content_pred_list):
        recall, ndcg = [], []
        for k in [10, 50]:
            recall.append(recall_at_k(answers, content_pred_list, k))
            ndcg.append(ndcg_k(answers, content_pred_list, k))
        content_scores = [recall[0], ndcg[0], recall[1], ndcg[1]]
        post_fix = {
            "Epoch": epoch,
            "content_HIT@10": '{:.4f}'.format(content_scores[0]),
            "content_NDCG@10": '{:.4f}'.format(content_scores[1]),
            "content_HIT@50": '{:.4f}'.format(content_scores[2]),
            "content_NDCG@50": '{:.4f}'.format(content_scores[3]),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return content_scores, str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def sequence_cross_entropy(self, seq_out, pos_ids, item_embeddings):
        logits = torch.matmul(seq_out, item_embeddings.weight.transpose(0, 1))
        logits = logits.view(-1, logits.size(-1))
        labels = pos_ids.view(-1)
        return F.cross_entropy(logits, labels, ignore_index=0)

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            id_ce_avg_loss = 0.0
            content_ce_avg_loss = 0.0
            align_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, rec_batch in rec_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                if getattr(self.model, "is_dual_branch", False):
                    id_sequence_output, content_sequence_output = self.model.encode_branches(input_ids)
                    id_ce_loss = self.sequence_cross_entropy(
                        id_sequence_output,
                        target_pos,
                        self.model.id_item_embeddings,
                    )
                    content_ce_loss = self.sequence_cross_entropy(
                        content_sequence_output,
                        target_pos,
                        self.model.content_item_embeddings,
                    )
                    align_loss = self.model.alignment_loss(
                        id_sequence_output,
                        content_sequence_output,
                        input_ids,
                    )
                    rec_loss = (
                        self.args.id_ce_weight * id_ce_loss
                        + self.args.content_ce_weight * content_ce_loss
                        + self.args.align_weight * align_loss
                    )
                    id_ce_avg_loss += id_ce_loss.item()
                    content_ce_avg_loss += content_ce_loss.item()
                    align_avg_loss += align_loss.item()
                else:
                    sequence_output = self.model.transformer_encoder(input_ids)
                    rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                self.optim.zero_grad(set_to_none=True)
                rec_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
            }
            if getattr(self.model, "is_dual_branch", False):
                post_fix["id_ce_loss"] = '{:.4f}'.format(id_ce_avg_loss / len(rec_data_iter))
                post_fix["content_ce_loss"] = '{:.4f}'.format(content_ce_avg_loss / len(rec_data_iter))
                post_fix["align_loss"] = '{:.4f}'.format(align_avg_loss / len(rec_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None
            content_pred_list = None

            if full_sort:
                answer_list = None
                content_item_bank = None
                if getattr(self.model, "is_dual_branch", False):
                    with torch.no_grad():
                        content_item_bank = self.model.content_item_embeddings.weight
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    with torch.no_grad():
                        if getattr(self.model, "is_dual_branch", False):
                            content_sequence_output = self.model.encode_content_branch(input_ids)
                            content_output = content_sequence_output[:, -1, :]
                            content_rating_pred = torch.matmul(
                                content_output,
                                content_item_bank.transpose(0, 1),
                            )
                        else:
                            recommend_output = self.model.transformer_encoder(input_ids)
                            recommend_output = recommend_output[:, -1, :]
                            rating_pred = self.predict_full(recommend_output)

                    batch_user_index = user_ids.cpu().numpy()

                    if getattr(self.model, "is_dual_branch", False):
                        content_rating_pred = content_rating_pred.cpu().data.numpy().copy()
                        seen_mask = self.args.train_matrix[batch_user_index].toarray() > 0
                        content_rating_pred[seen_mask] = 0

                        content_ind = np.argpartition(content_rating_pred, -50)[:, -50:]
                        content_arr_ind = content_rating_pred[np.arange(len(content_rating_pred))[:, None], content_ind]
                        content_arr_ind_argsort = np.argsort(content_arr_ind)[np.arange(len(content_rating_pred)), ::-1]
                        batch_content_pred_list = content_ind[np.arange(len(content_rating_pred))[:, None], content_arr_ind_argsort]

                        if i == 0:
                            content_pred_list = batch_content_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            content_pred_list = np.append(content_pred_list, batch_content_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    else:
                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                        ind = np.argpartition(rating_pred, -50)[:, -50:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

                if getattr(self.model, "is_dual_branch", False):
                    return self.get_content_branch_full_sort_score(
                        epoch,
                        answer_list,
                        content_pred_list,
                    )
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    with torch.no_grad():
                        recommend_output = self.model.finetune(input_ids)
                        test_neg_items = torch.cat((answers, sample_negs), -1)
                        recommend_output = recommend_output[:, -1, :]
                        test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
