import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


def pgd_untar(model, x_nat, y, num_steps, step_size, epsilon):
    '''
    calculate and return untargeted PGD attacked versions of the images in x_nat
    '''
    model.eval()
    ## add uniform noise
    inputs = x_nat.clone()
    ad_inputs = inputs + torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).to(x_nat.device)
    for _ in range(num_steps):

      ## calculate the gradient
      ad_inputs.requires_grad = True
      criterion = nn.CrossEntropyLoss()
      outputs = model(ad_inputs)
      loss = criterion(outputs, y)
      loss.backward()

      ## add the scaled signed gradient
      delta = step_size * torch.sign(ad_inputs.grad)
      ad_inputs = ad_inputs.detach() # detach from the graph
      ad_inputs += delta 

      ## project the pertubation
      ad_inputs = inputs + (ad_inputs - inputs).clamp(-epsilon, epsilon) 

      ## clamp to valid inputs
      torch.clamp(ad_inputs, min=0, max=1)

    return ad_inputs

# def pgd_untar(model, x_nat, y, num_steps, step_size, epsilon):
#     '''
#     calculate and return untargeted PGD attacked versions of the images in x_nat
#     '''
#     X_pgd = Variable(x_nat.data, requires_grad=True)
#     random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(x_nat.device)
#     X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

#     for _ in range(num_steps):
#         opt = optim.SGD([X_pgd], lr=1e-3)
#         opt.zero_grad()

#         with torch.enable_grad():
#             loss = nn.CrossEntropyLoss()(model(X_pgd), y)
#         loss.backward()
#         eta = step_size * X_pgd.grad.data.sign()
#         X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
#         eta = torch.clamp(X_pgd.data - x_nat.data, -epsilon, epsilon)
#         X_pgd = Variable(x_nat.data + eta, requires_grad=True)
#         X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        
#     return X_pgd

def pgd_tar(model, x_nat, y, num_steps, step_size, epsilon):
    '''calculate and return targeted PGD attacked versions of the images in x_nat'''

    model.eval()

    ## add uniform noise
    inputs = x_nat.clone()
    ad_inputs = inputs + torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).to(x_nat.get_device())
    tar_y = (y + 1) % 10
    for _ in range(num_steps):

      ## calculate the gradient
      ad_inputs.requires_grad = True
      criterion = nn.CrossEntropyLoss()
      outputs = model(ad_inputs)
      loss = criterion(outputs, tar_y)
      loss.backward()

      ## add the scaled signed gradient
      delta = step_size * torch.sign(ad_inputs.grad)
      ad_inputs = ad_inputs.detach() # detach from the graph
      ad_inputs -= delta 

      ## project the pertubation
      ad_inputs = inputs + (ad_inputs - inputs).clamp(-epsilon, epsilon) 

      ## clamp to valid inputs
      torch.clamp(ad_inputs, min=0, max=1)

    return ad_inputs

def fgsm_untar(model, x_nat, y, step_size):
    '''calculate and return untargeted FGSM attacked versions of the images in x_nat'''

    model.eval()
    inputs = x_nat.clone()
    inputs.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = criterion(outputs, y)
    loss.backward()

    delta = step_size * torch.sign(inputs.grad)

    inputs = inputs.detach()
    ad_inputs = inputs + delta
    torch.clamp(ad_inputs, min=0.0, max=1.0)

    return ad_inputs

def fgsm_tar(model, x_nat, y, step_size):
    '''calculate and return targeted FGSM attacked versions of the images in x_nat'''
    model.eval()

    inputs = x_nat.clone()
    inputs.requires_grad = True

    tar_y = (y + 1) % 10

    criterion = nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = criterion(outputs, tar_y)
    loss.backward()

    delta = step_size * torch.sign(inputs.grad)
    inputs = inputs.detach()
    ad_inputs = inputs - delta
    torch.clamp(ad_inputs, min=0.0, max=1.0)

    return ad_inputs

_ATTACK_FACTORY = {
   "PGD": {
        "untar": pgd_untar,
        "tar": pgd_tar
   },
   "FGSM":{
        "untar": fgsm_untar,
        "tar": fgsm_tar
   }
}

def ATTACK_FACTORY(type, tar, kwargs):
   return _ATTACK_FACTORY[type][tar](**kwargs)