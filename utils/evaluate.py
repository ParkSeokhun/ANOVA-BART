# https://github.com/ggong369/mBNN/blob/main/evaluate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import calibration as cal
from netcal.metrics import ECE
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def ll_mixture_normal(output, target, sigma):        
    exponent = -((target - output)**2).T/(2 * sigma**2)                     # (s, n) -> (n, s) / (s) -> (n, s)
    log_coeff = -0.5*torch.log(2*torch.tensor(np.pi))-torch.log(sigma)      # (s,)
    px = torch.mean(torch.exp(exponent + log_coeff),1)                      # (n, s) -> (n,) : likelihood의 sample-wise mean
    ll = torch.where(px!=0, torch.log(px), torch.mean(exponent + log_coeff,1))      # (n,)
    return torch.sum(ll)

def A(mu, sigma2):
    sigma = torch.sqrt(sigma2)
    r = (mu/sigma).detach().cpu().numpy()    
    # A1 = 2*sigma*(torch.from_numpy(norm.pdf(r)).float().cuda())
    A1 = 2*sigma*(torch.from_numpy(norm.pdf(r)).float())
    A2 = mu*(torch.from_numpy(2*norm.cdf(r)-1).float())    
    return(A1 + A2)

def CRPS_mixnorm(w,mu,sigma2,x):
    M = len(w)
    if (len(mu)!=M or len(sigma2)!=M): return(None)
    if x.dim()>0 :
        if len(x)>1:
            return(None)
    w = w/torch.sum(w)     
    crps1 = torch.sum(w*A(x-mu, sigma2))    
    crps3=[]
    for m in range(M):
        crps3.append(torch.sum(w*A(mu[m]-mu,sigma2[m] + sigma2)))    
    crps3 = torch.stack(crps3)
    crps2 = torch.sum(crps3*w/2)    
    return crps1 - crps2

def CRPS_norm(mu,sigma2,x):    
    if x.dim()>0 :
        if len(x)>1:
            return(None)
    crps1 = A(x-mu, sigma2)    
    crps2 = 0.5*A(0,2*sigma2)    
    return crps1 - crps2

def evaluate_averaged_model_regression(pred_list, target_list, sigma_list):
    """
    pred_list : torch.Tensor (#samples, n)
    target_list : torch.Tensor (n,)
    sigma_list : torch.Tensor (#samples,) - sd. NOT variance!
    """
    # CRPS_list=[]
    # for i in range(len(target_list)):
        # CRPS = CRPS_mixnorm(torch.ones(pred_list.shape[0]).cuda(), pred_list[:,i], sigma_list**2, target_list[i])
    #     CRPS = CRPS_mixnorm(torch.ones(pred_list.shape[0]), pred_list[:,i], sigma_list**2, target_list[i])
    #     CRPS_list.append(CRPS)
    # CRPSs = torch.stack(CRPS_list)  
    num_samples = pred_list.shape[0]
    abs_mean_value = torch.mean(np.abs(pred_list - target_list.reshape(1, -1)), dim = 0).squeeze()        # (s,n)-(1,n)->(n,1)->(n,)
    mean_value = torch.mean(pred_list, dim=0).squeeze()                    # (s, n) -> (n,)
    quantile = torch.arange(num_samples).reshape(-1, 1) / num_samples      # (s, 1) : sorted empirical cdf 값에 해당
    sorted_forward, _ = torch.sort(pred_list, dim = 0)                        # (s, n) -> (s, n) : sorted prediction
    quantile_mean = torch.mean( sorted_forward * quantile, dim = 0).squeeze()         # (s, n) -> (n,)
    crps = torch.mean(abs_mean_value + mean_value - 2 * quantile_mean).item()         # (n,) -> (1,) -> float

    RMSE = torch.sqrt(((torch.mean(pred_list,0) - target_list)**2).mean()).item()
    m_NLL = -ll_mixture_normal(pred_list, target_list, sigma_list).item() / pred_list.shape[1]
    # CRPS = torch.mean(CRPSs).item()    
    return(RMSE, m_NLL, crps)

def evaluate_averaged_model_classification(pred_list, target_list):
    """
    pred_list : probability torch.Tensor (#samples, n, #class) - NOT logit
    target_list : torch.Tensor (n,)
    """
    target_list = target_list.long()                        # int64로 변경
    outputs_mixture = torch.mean(pred_list, dim=0)          # (n, #class) - BMA result
    ACC= torch.mean((torch.argmax(outputs_mixture,1) == target_list).float()).item()

    criterion = torch.nn.NLLLoss(reduction='mean')
    if outputs_mixture.shape[1] == 1:
        outputs_mixture_ = torch.cat([1-outputs_mixture, outputs_mixture], dim = 1)     # binary인 경우 (n, 2)로 바꿔줌 -> NLL loss 계산 시 필요
    else :
        outputs_mixture_ = outputs_mixture.clone()
    m_NLL = criterion(torch.log(outputs_mixture_), target_list).item()
    
    if outputs_mixture.shape[1] >= 2:
        ece = cal.get_calibration_error(outputs_mixture.detach().cpu().numpy(), target_list.detach().cpu().numpy())
        return (ACC, m_NLL, ece)
    else :
        auroc = roc_auc_score(target_list.detach().cpu().numpy(), outputs_mixture.detach().cpu().squeeze().numpy())
        ece_msr = ECE(bins=20)
        ece = ece_msr.measure(outputs_mixture.detach().cpu().squeeze().numpy(), target_list.detach().cpu().numpy())
        return(ACC, auroc, m_NLL, ece)